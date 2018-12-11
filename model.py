import torch
from ops import *
from torch.autograd import Variable




class EncoderBlock(nn.Module):
    """
    A helper class that performs 2 convlutions, 1 MaxPool and 1 batch norm
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, pooling=True):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.kernel_size = kernel_size

        self.conv1 = conv(self.in_channels, self.out_channels, self.kernel_size)
        self.conv2 = conv(self.out_channels, self.out_channels, self.kernel_size)

        if self.pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.batch_norm = nn.BatchNorm2d(self.out_channels)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.max_pool(x)

        x = self.batch_norm(x)

        return x, before_pool


class DecoderBlock(nn.Module):
    """
    A helper class that performs 2 convlutions, 1 up convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 merge_mode='concat', up_mode='transpose'):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv2x2 = up_conv(self.in_channels, self.out_channels, kernel_size=2, mode=self.up_mode)
        self.upconv3x3 = up_conv(self.in_channels, self.out_channels, kernel_size=3, mode=self.up_mode)

        if self.merge_mode == 'concat':
            # number of input channel for conv1 is twice the out channels
            self.conv1 = conv(2*self.out_channels, self.out_channels, kernel_size)
        else:
            # number of input channel for conv1 is the same as the out channels
            self.conv1 = conv(self.out_channels, self.out_channels, kernel_size)

        self.conv2 = conv(self.out_channels, self.out_channels, kernel_size)

    def forward(self, from_encoder, to_decoder):

        if from_encoder.shape[2] % 2 == 0:
            to_decoder = self.upconv2x2(to_decoder)
        else:
            to_decoder = self.upconv3x3(to_decoder)

        if self.merge_mode == 'concat':
            # concating by channel
            x = torch.cat((to_decoder, from_encoder), 1)
        else:
            x = to_decoder + from_encoder

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x


class UNet(nn.Module):

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filt_num=64, filt_num_factor=2, up_mode='transpose', merge_mode='concat'):
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("{} is not a valid mode for upsampling. "
                             "Only \"transpose\" and \"upsample\" are allowed".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("{} is not a valid mode for merging encoder and decoder paths "
                             "only \"concat\" and \"add\" are valid.")

        if self.up_mode == 'upsample' and self.merge_mode=='add':
            raise ValueError("up_mode \"upsample\" is incompatible with merge_mode \"add\". ")

        self.in_channels = in_channels
        self.start_filt_num = start_filt_num
        self.depth = depth
        self.num_classes = num_classes
        self.filt_num_factor = filt_num_factor

        self.encoder_path = []
        self.decoder_path = []

        out_filters = self.in_channels

        # encoder path
        for i in range(depth):
            in_filters = out_filters
            out_filters = self.start_filt_num*(self.filt_num_factor**i)
            pooling = True if i < depth-1 else False
            enc_block = EncoderBlock(in_filters, out_filters, pooling=pooling)
            self.encoder_path.append(enc_block)

        # decoder path
        for i in range(depth-1):
            in_filters = out_filters
            out_filters = in_filters // filt_num_factor
            decoder_block = DecoderBlock(in_filters, out_filters,
                                         merge_mode=self.merge_mode,
                                         up_mode=self.up_mode)

            self.decoder_path.append(decoder_block)

        self.ending_conv = conv(out_filters, self.num_classes, kernel_size=1, padding=0)

        #TODO using ModuleDict with Encoder and Decoder as keys
        self.encoder_module = nn.ModuleList(self.encoder_path)
        self.decoder_module = nn.ModuleList(self.decoder_path)

        initialize_weights(self)
        # print(self.encoder_module)
        # print(self.decoder_module)

    def forward(self, x):
        encoder_outs = []

        for i, module in enumerate(self.encoder_module):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.decoder_module):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        x = self.ending_conv(x)
        return x


if __name__ == "__main__":

    model = UNet(num_classes=3, in_channels=1, depth=5, filt_num_factor=2, merge_mode='concat')

    model.apply(weight_init)
    cuda = torch.cuda.is_available()
    x = Variable(torch.rand(1, 1, 584, 584))
    if cuda:
        model.cuda()
        x = x.cuda()
    with torch.no_grad():
        out = model(x)
        print(out.shape)