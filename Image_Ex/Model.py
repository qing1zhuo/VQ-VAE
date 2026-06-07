import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
"""
向量量化（码本）类
初始化嵌入层
前向传播约定给一个CHW的图像张量
"""
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,     # 码本中“词条数”K
        embedding_dim,      # 码本维度D
        commitment_cost,    # 承诺损失参数
        decay,              # 指数滑动平均历史保留系数
        eps=1e-5            # eps防止除0
    ):
        super().__init__()
        # 保存参数
        self._commitment_cost=commitment_cost
        self._num_embeddings=num_embeddings
        self._embedding_dim=embedding_dim
        self._decay = decay
        self._eps = eps

        # 嵌入层
        self._embedding=nn.Embedding(self._num_embeddings,self._embedding_dim)
        nn.init.normal_(self._embedding.weight)
        self._embedding.weight.requires_grad_(False)

        # 使用次数的滑动平均，注册为标量，不参与优化更新 (K,)
        self.register_buffer('_ema_cluster_size',torch.zeros(num_embeddings))
        # 历史向量的滑动平均 (K,D)，注册为标量，不参与优化更新
        self.register_buffer('_ema_w', torch.zeros(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

    def forward(self,img):
        # TODO 1: 形状预处理
        # BCHW->BHWC
        img=img.permute(0,2,3,1).contiguous()
        img_shape=img.shape
        # 展平为(BHW,C)/(BHW,D)
        flat_img=img.view(-1,self._embedding_dim)

        # TODO 2: 计算距离, 得到形状为(BHW, K)的张量
        distances=(
            torch.sum(flat_img**2,dim=1,keepdim=True)
            +torch.sum(self._embedding.weight**2,dim=1)
            -2*(flat_img@self._embedding.weight.T)
        )

        # TODO 3: 计算索引和对应的隐向量
        # 找索引, 形状为(BHW, 1)
        indices=torch.argmin(distances,dim=1).unsqueeze(1)

        # 找对应的向量上，使用独热编码绕一圈，方便做使用率统计
        # 做一个独热矩阵(N,K)
        # 先做全零矩阵
        encodings=torch.zeros(indices.shape[0],self._num_embeddings,device=img.device)
        # 在列上按照index给出的索引写1
        encodings.scatter_(1,indices,1)

        # 得到对应的向量(BHW,D)
        quantized=encodings@self._embedding.weight
        # 形状变为(B,H,W,C)
        quantized=quantized.view(img_shape)

        # TODO 4: 更新码本
        if self.training:
            # 更新K个码本向量各自的移动平均使用次数
            self._ema_cluster_size=self._ema_cluster_size*self._decay+(1-self._decay)*torch.sum(encodings,dim=0)
            # 计算当前总的使用次数
            n=torch.sum(self._ema_cluster_size.data)
            # Laplace平滑，让那些没怎么使用的向量对应的使用次数不至于为0
            self._ema_cluster_size=(self._ema_cluster_size+self._eps)/(n+self._num_embeddings*self._eps)*n

            # (K,D) 每个码本向量被分配到的latent向量进行元素级求和得到的东西
            dw=encodings.T@flat_img   
            # 更新累加向量的移动平均（detach 切断梯度，EMA 不参与反向传播）
            self._ema_w=self._ema_w*self._decay+(1-self._decay)*dw.detach()

            # 更新码本 (K,D)/(K,1) — 直接原地更新，不创建新 Parameter
            self._embedding.weight.data.copy_(self._ema_w/self._ema_cluster_size.unsqueeze(1))

        # TODO 5: 计算承诺损失
        # 把quantized从计算图剥离，避免承诺损失梯度回传到码本
        e_latene_loss=F.mse_loss(quantized.detach(),img)
        commit_loss=self._commitment_cost*e_latene_loss

        # TODO 6: 直通梯度传递
        quantized=img+(quantized-img).detach()

        # TODO 7: 计算码本平均使用率
        # (K,)
        avg_probs=encodings.mean(dim=0)
        # 等效有几个 code 在被使用
        perplexity=torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return commit_loss,perplexity,quantized.permute(0,3,1,2).contiguous(),encodings

'''
残差连接块
输入维度->残差维度->隐藏向量维度
结构: relu + 3*3 conv + relu + 1*1 conv
'''
class Residual(nn.Module):
    def __init__(self,in_channels,residual_channels,out_channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels,residual_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True),
            nn.Conv2d(residual_channels,out_channels,kernel_size=1,stride=1)
        )
    def forward(self,x):
        return x+self.block(x)
'''
残差模块，由两个残差块组成
'''
class ResidualStack(nn.Module):
    def __init__(self,in_channels,residual_channels,out_channels,residual_layers):
        super().__init__()
        self._residual_layers=residual_layers
        self._layers=nn.ModuleList(
            [Residual(in_channels,residual_channels,out_channels) 
            for _ in range(self._residual_layers)]
            )
    def forward(self,x):
        for i in range(self._residual_layers):
            x=self._layers[i](x)
        return F.relu(x)

'''
编码器encoder
'''
class Encoder(nn.Module):
    def __init__(
        self,in_channels,hidden_channels,residual_channels,residual_layers):
        super().__init__()
        # (in_c,32,32)->(hid_c//2,16,16)
        self.conv1=nn.Conv2d(
            in_channels,hidden_channels//2,
            kernel_size=4,stride=2,padding=1
            )
        # (hid_c//2,16,16)->(hid_c,8,8)
        self.conv2=nn.Conv2d(
            hidden_channels//2,hidden_channels,
            kernel_size=4,stride=2,padding=1
            )
        # (hid_c,8,8)->(hid_c,8,8)
        # 第三层卷积对特征做一个处理再去接relu
        self.conv3=nn.Conv2d(
            hidden_channels,hidden_channels,
            kernel_size=3,stride=1,padding=1
            )  
        # (hid_c,8,8)->(hid_c,8,8)
        self.residual_stack=ResidualStack(hidden_channels,residual_channels,hidden_channels,residual_layers)
    def forward(self,x):
        out=F.relu(self.conv1(x))
        out=F.relu(self.conv2(out))
        out=self.conv3(out)
        return self.residual_stack(out)
'''
解码器decoder
'''
class Decoder(nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,residual_channels,residual_layers):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,hidden_channels,kernel_size=3,stride=1,padding=1)

        self.residual_stack=ResidualStack(hidden_channels,residual_channels,hidden_channels,residual_layers)

        self.conv2=nn.ConvTranspose2d(hidden_channels,hidden_channels//2,kernel_size=4,stride=2,padding=1)
        
        self.conv3=nn.ConvTranspose2d(hidden_channels//2,out_channels,kernel_size=4,stride=2,padding=1)
    def forward(self,x):
        out=self.conv1(x)
        out=self.residual_stack(out)
        out=F.relu(self.conv2(out))
        return self.conv3(out)
'''
完整模型
'''
class Model(nn.Module):
    def __init__(
        self,
        in_channels,hidden_channels,out_channels,
        residual_channels,residual_layers,
        num_embeddings,embedding_dim,decay,commitment_cost
    ):
        super().__init__()
        self.encoder=Encoder(in_channels,hidden_channels,residual_channels,residual_layers)

        self.in_vq_conv=nn.Conv2d(hidden_channels,embedding_dim,1,1)
        
        self.vq=VectorQuantizerEMA(num_embeddings,embedding_dim,commitment_cost,decay)

        self.decoder=Decoder(embedding_dim,hidden_channels,out_channels,residual_channels,residual_layers)
    def forward(self,x):
        z_e=self.in_vq_conv(self.encoder(x))
        commit_loss,perplexity,z_q,_=self.vq(z_e)
        x_recon=self.decoder(z_q)
        return commit_loss,perplexity,x_recon

    def encode(self,image):
        # 输入image (B,C,H,W)
        # 首先生成了编码后的基本特征图
        z_e=self.in_vq_conv(self.encoder(image))
        B,C,H,W=z_e.shape
        # 给出(n,k)的独热编码索引本
        _,_,_,encodings=self.vq(z_e)
        # 变为索引本(n,)
        indices=torch.argmax(encodings,dim=1)
        # reshape回(B,H,W)
        return indices.view(B,H,W)

    def decode(self,indices):
        # 输入(B,H,W)索引码本
        z_q=self.vq._embedding(indices) # (B,H,W,D)
        z_q=z_q.permute(0,3,1,2).contiguous()  # (B,D,H,W)
        return self.decoder(z_q)
        
class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)  # 按照特征切为两半 (N,2D)->2(N,D)
        return F.tanh(x1)*F.sigmoid(x2)
        
class GatedMaskConv(nn.Module):
    @staticmethod
    def _causal_masks(mask_type, vert_weight_shape, horiz_weight_shape):
        """Mask-A: 垂直核末行、水平核末列置零；Mask-B: 全 1（与原先隐式几何一致）。"""
        v_mask = torch.ones(vert_weight_shape)
        h_mask = torch.ones(horiz_weight_shape)
        if mask_type == "A":
            v_mask[:, :, -1, :] = 0
            h_mask[:, :, :, -1] = 0
        return v_mask, h_mask

    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        self._residual=residual
        self._mask_type=mask_type

        # 嵌入层，把(B,)->(B,2D)
        self._embedding=nn.Embedding(n_classes,2*dim)

        # TODO 1 : 垂直卷积
        kernel_shp=(kernel//2+1,kernel)
        padding_shp=(kernel//2,kernel//2)
        self._vertic_conv=nn.Conv2d(    #(in_c,out_c,kernel//2+1,kernel)
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=kernel_shp,
            stride=1,
            padding=padding_shp
        )
        
        self._vertic_horiz=nn.Conv2d(2*dim,2*dim,1)

        # TODO 2 : 水平卷积
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self._horiz_conv = nn.Conv2d(   #(in_c,out_c,1,kernel//2+1)
            in_channels=dim,
            out_channels=2*dim,
            kernel_size=kernel_shp,
            stride=1,
            padding=padding_shp
        )

        self._horiz_resid=nn.Conv2d(dim,dim,1)

        self._gate=GatedActivation()

        v_mask, h_mask = self._causal_masks(
            mask_type,
            self._vertic_conv.weight.shape,
            self._horiz_conv.weight.shape,
        )
        self.register_buffer("_v_mask", v_mask)
        self.register_buffer("_h_mask", h_mask)

    def _masked_conv2d(self, x, conv, mask):
        return F.conv2d(
            x,
            conv.weight * mask,
            conv.bias,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    def forward(self,x_v,x_h,h):
        # x_v (B,D,H,W), x_h (B,D,H,W)
        h_embed=self._embedding(h)  # (B,2D)
        h_vertic=self._masked_conv2d(x_v, self._vertic_conv, self._v_mask)  # (B,2D,H,W)
        h_vertic=h_vertic[:,:,:x_v.size(-2),:]
        out_v=self._gate(h_vertic+h_embed[:,:,None,None])   # (B,D,H,W)

        h_horiz=self._masked_conv2d(x_h, self._horiz_conv, self._h_mask)
        h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
        v2h = self._vertic_horiz(h_vertic)

        out_h = self._gate(v2h + h_horiz + h_embed[:, :, None, None])

        if self._residual:
            out_h=self._horiz_resid(out_h)+x_h
        else:
            out_h=self._horiz_resid(out_h)

        return out_v,out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=512, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # 把离散索引嵌入成连续向量 (B, H, W) -> (B, H, W, dim)
        self.embedding = nn.Embedding(input_dim, dim)

        # 依次堆叠 GatedMaskConv，第一层 Mask-A，其余 Mask-B
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskConv(mask_type, dim, kernel, residual, n_classes)
            )

        # 把每个空间位置的特征 dim -> input_dim(K)，每个位置上一个 K 维 logits
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, input_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(input_dim, input_dim, 1)
        )

    def forward(self, x, label):
        # x: (B, H, W) int64, label: (B,) int64
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)   # (B, input_dim, H, W)

    def generate(self, label, shape=(8, 8), batch_size=64, temperature=1.0):
        with torch.no_grad():
            self.eval()
            param = next(self.parameters())
            x = torch.zeros(
                (batch_size, *shape),
                dtype=torch.int64, device=param.device
            )
            label=label.to(param.device)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    logits = self.forward(x, label) # (B, input_dim, H, W)
                    logits = logits[:, :, i, j] / temperature
                    probs = F.softmax(logits, -1)
                    x.data[:, i, j].copy_(
                        torch.multinomial(probs, 1).squeeze().data
                    )
        return x


