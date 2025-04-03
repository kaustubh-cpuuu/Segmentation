import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, rate=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=rate, dilation=rate),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class RSU_L(nn.Module):
    def __init__(self, in_c, out_c, int_c, num_layers, rate=2):
        super().__init__()

        """ Initial Conv """
        self.c1 = conv_block(in_c, out_c)

        """ Encoder """
        self.c2 = conv_block(out_c, int_c)

        self.c3 = nn.ModuleList()
        for i in range(num_layers - 2):
            conv = nn.Sequential(
                nn.MaxPool2d((2, 2)),
                conv_block(int_c, int_c)
            )
            self.c3.append(conv)

        """ Bridge """
        self.c4 = conv_block(int_c, int_c, rate=rate)

        """ Decoder """
        self.c5 = conv_block(int_c*2, int_c)

        self.c6 = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        for i in range(num_layers - 3):
            self.c6.append(conv_block(int_c*2, int_c))

        self.c7 = conv_block(int_c*2, out_c)

    def forward(self, inputs):
        """ Initial Conv """
        x = self.c1(inputs)
        init_feats = x

        """ Encoder """
        skip = []
        x = self.c2(x)
        skip.append(x)

        for i in range(len(self.c3)):
            x = self.c3[i](x)
            skip.append(x)

        """ Bridge """
        x = self.c4(x)

        """ Decoder """
        skip.reverse()

        x = torch.cat([x, skip[0]], dim=1)
        x = self.c5(x)

        for i in range(len(self.c6)):
            x = self.up(x)
            x = torch.cat([x, skip[i+1]], dim=1)
            x = self.c6[i](x)

        x = self.up(x)
        x = torch.cat([x, skip[-1]], dim=1)
        x = self.c7(x)

        """ Add """
        x = x + init_feats
        return x

class RSU_4F(nn.Module):
    def __init__(self, in_c, out_c, int_c):
        super().__init__()

        """ Initial Conv """
        self.c1 = conv_block(in_c, out_c, rate=1)

        """ Encoder """
        self.c2 = conv_block(out_c, int_c, rate=1)
        self.c3 = conv_block(int_c, int_c, rate=2)
        self.c4 = conv_block(int_c, int_c, rate=4)

        """ Bridge """
        self.c5 = conv_block(int_c, int_c, rate=8)

        """ Decoder """
        self.c6 = conv_block(int_c*2, int_c, rate=4)
        self.c7 = conv_block(int_c*2, int_c, rate=2)
        self.c8 = conv_block(int_c*2, out_c, rate=1)

    def forward(self, inputs):
        """ Initial Conv """
        x1 = self.c1(inputs)

        """ Encoder """
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)

        """ Bridge """
        x5 = self.c5(x4)

        """ Decoder """
        x = torch.cat([x5, x4], dim=1)
        x = self.c6(x)

        x = torch.cat([x, x3], dim=1)
        x = self.c7(x)

        x = torch.cat([x, x2], dim=1)
        x = self.c8(x)

        """ Add """
        x = x + x1
        return x

class u2net(nn.Module):
    def __init__(self, in_c, out_c, int_c, num_classes):
        super().__init__()

        """ Encoder """
        self.pool = nn.MaxPool2d((2, 2))
        self.s1 = RSU_L(in_c[0], out_c[0], int_c[0], 7)
        self.s2 = RSU_L(in_c[1], out_c[1], int_c[1], 6)
        self.s3 = RSU_L(in_c[2], out_c[2], int_c[2], 5)
        self.s4 = RSU_L(in_c[3], out_c[3], int_c[3], 4)
        self.s5 = RSU_4F(in_c[4], out_c[4], int_c[4])

        """ Bridge """
        self.b1 = RSU_4F(in_c[5], out_c[5], int_c[5])
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        """ Decoder """
        self.s6 = RSU_4F(in_c[6], out_c[6], int_c[6])
        self.s7 = RSU_L(in_c[7], out_c[7], int_c[7], 4)
        self.s8 = RSU_L(in_c[8], out_c[8], int_c[8], 5)
        self.s9 = RSU_L(in_c[9], out_c[9], int_c[9], 6)
        self.s10 = RSU_L(in_c[10], out_c[10], int_c[10], 7)

        """ Side Outputs """
        self.y1 = nn.Conv2d(out_c[10], num_classes, kernel_size=3, padding=1)

        self.y2 = nn.Sequential(
            nn.Conv2d(out_c[9], num_classes, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.y3 = nn.Sequential(
            nn.Conv2d(out_c[8], num_classes, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )

        self.y4 = nn.Sequential(
            nn.Conv2d(out_c[7], num_classes, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        )

        self.y5 = nn.Sequential(
            nn.Conv2d(out_c[6], num_classes, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        )

        self.y6 = nn.Sequential(
            nn.Conv2d(out_c[5], num_classes, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True)
        )

        self.y0 = nn.Conv2d(6*num_classes, num_classes, kernel_size=3, padding=1)

    def forward(self, inputs):
        s0 = inputs

        """ Encoder """
        s1 = self.s1(s0)
        p1 = self.pool(s1)

        s2 = self.s2(p1)
        p2 = self.pool(s2)

        s3 = self.s3(p2)
        p3 = self.pool(s3)

        s4 = self.s4(p3)
        p4 = self.pool(s4)

        s5 = self.s5(p4)
        p5 = self.pool(s5)

        """ Bridge """
        b1 = self.b1(p5)
        b2 = self.up(b1)

        """ Decoder """
        d1 = torch.cat([b2, s5], dim=1)
        d1 = self.s6(d1)
        u1 = self.up(d1)

        d2 = torch.cat([u1, s4], dim=1)
        d2 = self.s7(d2)
        u2 = self.up(d2)

        d3 = torch.cat([u2, s3], dim=1)
        d3 = self.s8(d3)
        u3 = self.up(d3)

        d4 = torch.cat([u3, s2], dim=1)
        d4 = self.s9(d4)
        u4 = self.up(d4)

        d5 = torch.cat([u4, s1], dim=1)
        d5 = self.s10(d5)

        """ Side Outputs """
        y1 = self.y1(d5)
        y2 = self.y2(d4)
        y3 = self.y3(d3)
        y4 = self.y4(d2)
        y5 = self.y5(d1)
        y6 = self.y6(b1)

        y0 = torch.cat([y1, y2, y3, y4, y5, y6], dim=1)
        y0 = self.y0(y0)

        return y0, y1, y2, y3, y4, y5, y6


def build_u2net(num_classes=1):
    in_c = [3, 64, 128, 256, 512, 512, 1024, 1024, 512, 256, 128]
    out_c = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    int_c = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]
    return u2net(in_c, out_c, int_c, num_classes=num_classes)

def build_u2net_lite(num_classes=1):
    in_c = [3, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128]
    out_c = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    int_c = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    return u2net(in_c, out_c, int_c, num_classes=num_classes)

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = build_u2net()
    y0, y1, y2, y3, y4, y5, y6 = model(x)
    print(y0.shape, y1.shape, y2.shape, y3.shape, y4.shape, y5.shape, y6.shape)
