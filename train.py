import os, sys, time, pickle, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms

import utils
from models import gan_net
from models import s_net
from edge_promoting import edge_promoting

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='vangogh',  help='project name')
parser.add_argument('--train_epoch', type=int, default=200)
parser.add_argument('--pre_train_epoch', type=int, default=10)
parser.add_argument('--src_data', required=False, default='src_data_path',  help='path of source data')
parser.add_argument('--tgt_data', required=False, default='tgt_data_path',  help='path of target data')
parser.add_argument('--vgg_model', required=False, default='./vgg/vgg19.pth', help='path of pre-trained VGG19 model')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--ngf', type=int, default=64, help='the channel number of first conv block of generator')
parser.add_argument('--ndf', type=int, default=32, help='the channel number of first conv block of discriminator')
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate of generater, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate of discriminator, default=0.0002')
parser.add_argument('--lrS', type=float, default=0.0002, help='learning rate of saliency network, default=0.0002')
parser.add_argument('--content_weight', type=float, default=0.25, help='weight of content loss')
parser.add_argument('--content_weight_decay', type=float, default=0.96, help='decay rate for weight of content loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam optimizer')
parser.add_argument('--latest_generator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_discriminator_model', required=False, default='', help='the latest trained model path')
parser.add_argument('--latest_saliency_model', required=False, default='', help='the latest trained model path')
args = parser.parse_args()

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()

# results save path
if not os.path.isdir(os.path.join(args.name + '_results', 'Reconstruction')):
    os.makedirs(os.path.join(args.name + '_results', 'Reconstruction'))
if not os.path.isdir(os.path.join(args.name + '_results', 'Transfer')):
    os.makedirs(os.path.join(args.name + '_results', 'Transfer'))
if not os.path.isdir(os.path.join(args.name + '_results', 'model')):
    os.makedirs(os.path.join(args.name + '_results', 'model'))

# log file save path
sys.stdout = Logger(os.path.join(args.name + '_results', 'log.txt'))

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

print('cuda: %s' % torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

# edge-promoting
if not os.path.isdir(os.path.join('data', args.tgt_data, 'pair')):
    print('edge-promoting start!!')
    edge_promoting(os.path.join('data', args.tgt_data, 'train'), os.path.join('data', args.tgt_data, 'pair'))
else:
    print('edge-promoting already done')

# data_loader
src_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
tgt_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

train_loader_src = utils.data_load(os.path.join('data', args.src_data), 'train_com', tgt_transform, args.batch_size, shuffle=True, drop_last=True)
train_loader_tgt = utils.data_load(os.path.join('data', args.tgt_data), 'pair', tgt_transform, args.batch_size, shuffle=True, drop_last=True)
test_loader_src = utils.data_load(os.path.join('data', args.src_data), 'test_com', tgt_transform, 1, shuffle=True, drop_last=True)

# network
G = gan_net.Generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
if args.latest_generator_model != '':
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(args.latest_generator_model))
    else:
        G.load_state_dict(torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage))

D = gan_net.Discriminator(args.in_ndc, args.out_ndc, args.ndf)
if args.latest_discriminator_model != '':
    if torch.cuda.is_available():
        D.load_state_dict(torch.load(args.latest_discriminator_model))
    else:
        D.load_state_dict(torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage))

S = s_net.SaliencyNet(args.in_ngc, args.out_ngc, args.ngf, args.nb)
if args.latest_discriminator_model != '':
    if torch.cuda.is_available():
        S.load_state_dict(torch.load(args.latest_saliency_model))
    else:
        S.load_state_dict(torch.load(args.latest_saliency_model, map_location=lambda storage, loc: storage))

VGG = gan_net.VGG19(init_weights=args.vgg_model, feature_mode=True)

G.to(device)
D.to(device)
S.to(device)
VGG.to(device)
G.train()
D.train()
S.train()
VGG.eval()

print('---------- Networks initialized -------------')
utils.print_network(G)
utils.print_network(D)
utils.print_network(S)
utils.print_network(VGG)
print('-----------------------------------------------')

# loss
MSE_loss = nn.MSELoss().to(device)
L1_loss = nn.L1Loss().to(device)

# adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
S_optimizer = optim.Adam(S.parameters(), lr=args.lrS, betas=(args.beta1, args.beta2))
G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
S_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=S_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

# initialization phase
pre_train_hist = {}
pre_train_hist['Recon_loss'] = []
pre_train_hist['Sali_loss'] = []
pre_train_hist['per_epoch_time'] = []
pre_train_hist['total_time'] = []

if args.latest_generator_model == '':
    print('Pre-training start!')
    start_time = time.time()
    for epoch in range(args.pre_train_epoch):
        epoch_start_time = time.time()
        Recon_losses = []
        Sali_losses = []
        for x, _ in train_loader_src:
            x = x[:, [2, 1, 0], :, :]
            s = x[:, :, :, args.input_size:]
            x = x[:, :, :, :args.input_size]

            x = x.to(device)
            s = s.to(device)

            #train generator
            G_optimizer.zero_grad()

            x_feature = VGG((x + 1) / 2)
            G_ = G(x)
            G_feature = VGG((G_ + 1) / 2)

            Recon_loss = L1_loss(G_feature, x_feature.detach()) * args.content_weight
            Recon_losses.append(Recon_loss.item())
            pre_train_hist['Recon_loss'].append(Recon_loss.item())

            Recon_loss.backward()
            G_optimizer.step()

            #train saliency network
            S_optimizer.zero_grad()

            s_feature = S(x)[3]

            Sali_loss = L1_loss(s_feature, s.detach())
            Sali_losses.append(Sali_loss.item())
            pre_train_hist['Sali_loss'].append(Sali_loss.item())

            Sali_loss.backward()
            S_optimizer.step()

        per_epoch_time = time.time() - epoch_start_time
        pre_train_hist['per_epoch_time'].append(per_epoch_time)
        print('[%d/%d] - time: %.2f, Recon loss: %.3f, Sali loss: %.3f' % ((epoch), args.pre_train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses)), torch.mean(torch.FloatTensor(Sali_losses))))

    total_time = time.time() - start_time
    pre_train_hist['total_time'].append(total_time)
    with open(os.path.join(args.name + '_results', 'model', 'pre_train_hist.pkl'), 'wb') as f:
        pickle.dump(pre_train_hist, f)

    with torch.no_grad():
        G.eval()
        for n, (x, _) in enumerate(train_loader_src):
            s = x[:, :, :, args.input_size:]
            x = x[:, :, :, :args.input_size]
            x_bgr = x[:, [2, 1, 0], :, :]
            s = s.to(device)
            x = x.to(device)
            x_bgr = x_bgr.to(device)
            G_recon = G(x_bgr)
            S_gen = S(x_bgr)[3]
            G_recon_rgb = G_recon[:, [2, 1, 0], :, :]
            S_gen_rgb = S_gen[:, [2, 1, 0], :, :]
            result = torch.cat((x[0], G_recon_rgb[0]), 2)
            saliency = torch.cat((x[0], S_gen_rgb[0], s[0]), 2)
            path = os.path.join(args.name + '_results', 'Reconstruction',
                                args.name + '_train_recon_' + str(n + 1) + '.png')
            saliency_path = os.path.join(args.name + '_results', 'Reconstruction',
                                         args.name + '_train_saliency_' + str(n + 1) + '.png')

            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave(saliency_path, (saliency.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break

        for n, (x, _) in enumerate(test_loader_src):
            s = x[:, :, :, args.input_size:]
            x = x[:, :, :, :args.input_size]
            x_bgr = x[:, [2, 1, 0], :, :]
            s = s.to(device)
            x = x.to(device)
            x_bgr = x_bgr.to(device)
            G_recon = G(x_bgr)
            S_gen = S(x_bgr)[3]
            G_recon_rgb = G_recon[:, [2, 1, 0], :, :]
            S_gen_rgb = S_gen[:, [2, 1, 0], :, :]
            result = torch.cat((x[0], G_recon_rgb[0]), 2)
            saliency = torch.cat((x[0], S_gen_rgb[0], s[0]), 2)
            path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_test_recon_' + str(n + 1) + '.png')
            saliency_path = os.path.join(args.name + '_results', 'Reconstruction',
                                         args.name + '_test_saliency_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            plt.imsave(saliency_path, (saliency.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
            if n == 4:
                break
else:
    print('Load the latest generator model')

# training phase
train_hist = {}
train_hist['Disc_loss'] = []
train_hist['Gen_loss'] = []
train_hist['Con_loss'] = []
train_hist['Saliency_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []
print('training start!')
start_time = time.time()
real = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
fake = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
for epoch in range(args.train_epoch):
    epoch_start_time = time.time()
    G.train()
    S.train()
    G_scheduler.step()
    D_scheduler.step()
    S_scheduler.step()
    Disc_losses = []
    Gen_losses = []
    Con_losses = []
    Saliency_losses = []
    for (x, _), (y, _) in zip(train_loader_src, train_loader_tgt):
        x = x[:, [2, 1, 0], :, :]
        y = y[:, [2, 1, 0], :, :]
        s = x[:, :, :, args.input_size:]
        x = x[:, :, :, :args.input_size]
        e = y[:, :, :, args.input_size:]
        y = y[:, :, :, :args.input_size]
        x, y, e, s = x.to(device), y.to(device), e.to(device), s.to(device)

        # train discriminator
        D_optimizer.zero_grad()

        D_real = D(y)
        D_real_loss = MSE_loss(D_real, real)

        G_ = G(x)
        D_fake = D(G_)
        D_fake_loss = MSE_loss(D_fake, fake)

        D_edge = D(e)
        D_edge_loss = MSE_loss(D_edge, fake)

        Disc_loss = D_real_loss + D_fake_loss + D_edge_loss
        Disc_losses.append(Disc_loss.item())
        train_hist['Disc_loss'].append(Disc_loss.item())

        Disc_loss.backward()
        D_optimizer.step()

        # train generator
        G_optimizer.zero_grad()

        G_ = G(x)
        D_fake = D(G_)
        D_fake_loss = MSE_loss(D_fake, real)

        x_feature = VGG((x + 1) / 2)
        G_feature = VGG((G_ + 1) / 2)
        con_loss_decay = args.content_weight * args.content_weight_decay ** int(epoch/10)
        Con_loss = L1_loss(G_feature, x_feature.detach()) * con_loss_decay

        Gen_loss = D_fake_loss + Con_loss
        Gen_losses.append(D_fake_loss.item())
        train_hist['Gen_loss'].append(D_fake_loss.item())
        Con_losses.append(Con_loss.item())
        train_hist['Con_loss'].append(Con_loss.item())

        Gen_loss.backward()
        G_optimizer.step()

        # train salienct network
        S_optimizer.zero_grad()

        S_ = S(x)[3]

        Saliency_loss = L1_loss(S_, s.detach())
        Saliency_losses.append(Saliency_loss.item())
        train_hist['Saliency_loss'].append(Saliency_loss.item())

        Saliency_loss.backward()
        S_optimizer.step()

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)
    print(
    '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f，Saliency loss: %.3f' % ((epoch), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
        torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses)), torch.mean(torch.FloatTensor(Saliency_losses))))

    if epoch % 5 == 0 or epoch == args.train_epoch - 1:
        with torch.no_grad():
            G.eval()
            for n, (x, _) in enumerate(train_loader_src):
                s = x[:, :, :, args.input_size:]
                x = x[:, :, :, :args.input_size]
                x_bgr = x[:, [2, 1, 0], :, :]
                s = s.to(device)
                x = x.to(device)
                x_bgr = x_bgr.to(device)
                G_recon = G(x_bgr)
                S_gen = S(x_bgr)[3]
                G_recon_rgb = G_recon[:, [2, 1, 0], :, :]
                S_gen_rgb = S_gen[:, [2, 1, 0], :, :]
                result = torch.cat((x[0], G_recon_rgb[0]), 2)
                saliency = torch.cat((x[0], S_gen_rgb[0], s[0]), 2)
                path = os.path.join(args.name + '_results', 'Transfer',
                                    str(epoch) + '_epoch_' + args.name + '_train_' + str(n + 1) + '.png')
                saliency_path = os.path.join(args.name + '_results', 'Transfer',
                                    str(epoch) + '_epoch_' + args.name + '_train_saliency_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(saliency_path, (saliency.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 4:
                    break

            for n, (x, _) in enumerate(test_loader_src):
                s = x[:, :, :, args.input_size:]
                x = x[:, :, :, :args.input_size]
                x_bgr = x[:, [2, 1, 0], :, :]
                x_bgr = x_bgr.to(device)
                s = s.to(device)
                x = x.to(device)
                G_recon = G(x_bgr)
                S_gen = S(x_bgr)[3]
                G_recon_rgb = G_recon[:, [2, 1, 0], :, :]
                S_gen_rgb = S_gen[:, [2, 1, 0], :, :]
                result = torch.cat((x[0], G_recon_rgb[0]), 2)
                saliency = torch.cat((x[0], S_gen_rgb[0], s[0]), 2)
                path = os.path.join(args.name + '_results', 'Transfer', str(epoch) + '_epoch_' + args.name + '_test_' + str(n + 1) + '.png')
                saliency_path = os.path.join(args.name + '_results', 'Transfer',
                                    str(epoch) + '_epoch_' + args.name + '_test_saliency_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                plt.imsave(saliency_path, (saliency.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 4:
                    break

            torch.save(G.state_dict(), os.path.join(args.name + '_results', 'model', 'generator_epoch_' + str(epoch)))
            torch.save(D.state_dict(), os.path.join(args.name + '_results', 'model', 'discriminator_epoch_' + str(epoch)))
            torch.save(S.state_dict(), os.path.join(args.name + '_results', 'model', 'saliency_epoch_' + str(epoch)))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Average one epoch time: %.2f, total %d epoch time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish! Saving training results")

torch.save(G.state_dict(), os.path.join(args.name + '_results', 'model', 'generator_param.pkl'))
torch.save(D.state_dict(), os.path.join(args.name + '_results', 'model', 'discriminator_param.pkl'))
torch.save(D.state_dict(), os.path.join(args.name + '_results', 'model', 'saliency_param.pkl'))
with open(os.path.join(args.name + '_results', 'model', 'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)