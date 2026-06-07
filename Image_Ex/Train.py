import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import Model
import copy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ============================================================================
#                       ↓↓↓  图像 VQ-VAE 训练 / 验证  ↓↓↓
# ============================================================================

# 一轮训练
def train(
    model,train_loader,
    epoch,print_epoc,
    lr,wd,device,
    optimizer,criterion, # 损失计算应该是mse平均损失
    mean,std
):
    model.train()
    model=model.to(device)
    epoch_loss,samples=0.0,0
    epoch_recon_loss,epoch_commit_loss=0.0,0.0
    epoch_perplexity=0.0
    batch=0
    original_image,recon_image=None,None
    for x,_ in train_loader:
        x=x.to(device)

        commit_loss,perplexity,x_recon=model(x)
        recon_loss=criterion(x_recon,x)
        loss=recon_loss+commit_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_perplexity+=perplexity.item()*x.shape[0]
        epoch_commit_loss+=commit_loss.item()*x.shape[0]
        epoch_recon_loss+=recon_loss.item()*x.shape[0]
        epoch_loss+=loss.item()*x.shape[0]
        samples+=x.shape[0]

        if batch==0 and epoch%print_epoc==0:
            original_image=x
            recon_image=x_recon
            batch+=1

    epoch_perplexity/=samples
    epoch_loss/=samples
    epoch_recon_loss/=samples
    epoch_commit_loss/=samples

    if epoch%print_epoc==0:
        print(f"epoch: {epoch}")
        print(f"reconstruction loss={epoch_recon_loss}")
        print(f"commit loss={epoch_commit_loss}")
        print(f"total loss={epoch_loss}")
        print(f"perplexity={epoch_perplexity}\n")
        draw(original_image,recon_image,10,mean,std)

    return epoch_loss,epoch_recon_loss,epoch_commit_loss,epoch_perplexity

def train_pipeline(
    init_model,
    train_loader,
    epochs,print_epoc,lr,wd,device,
    mean,std
):
    best_model=None
    optimizer=optim.Adam(init_model.parameters(),lr,weight_decay=wd)
    criterion=nn.MSELoss()
    loss_list,recon_loss_list,commit_loss_list=[],[],[]
    perplexity_list=[]
    minn_loss=1e20

    for epoch in range(1,epochs+1):
        epoch_loss,epoch_recon_loss,epoch_commit_loss,epoch_perplexity=train(
            init_model,train_loader,epoch,print_epoc,lr,wd,device,optimizer,criterion,
            mean,std
        )
        loss_list.append(epoch_loss)
        recon_loss_list.append(epoch_recon_loss)
        commit_loss_list.append(epoch_commit_loss)
        perplexity_list.append(epoch_perplexity)

        if minn_loss>epoch_loss:
            minn_loss=epoch_loss
            best_model=copy.deepcopy(init_model)

    return perplexity_list,loss_list,recon_loss_list,commit_loss_list,best_model

def valid(model,valid_loader,device,mean,std,idx):
    model.eval()
    original_image,recon_image=None,None
    batch=0
    with torch.no_grad():
        for x,_ in valid_loader:
            batch+=1
            x=x.to(device)
            _,_,recon_x=model(x)
            original_image=x
            recon_image=recon_x
            if batch>=idx:
                break
    draw(original_image,recon_image,10,mean,std)

def draw(x,recon_x,n,mean,std):
    C = x.shape[1]  
    mean = torch.tensor(mean).view(1, C, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, C, 1, 1).to(x.device)
    
    n = min(x.shape[0],n)
    x_denorm = x * std + mean
    recon_x_denorm = recon_x * std + mean
    
    x_denorm = x_denorm.permute(0,2,3,1)[:10].cpu().detach().numpy()
    recon_x_denorm = recon_x_denorm.permute(0,2,3,1)[:10].cpu().detach().numpy()
    x_denorm = np.clip(x_denorm, 0, 1)  # 强制截断到[0,1]
    recon_x_denorm = np.clip(recon_x_denorm, 0, 1)
    plt.figure(figsize=(12,4))
    for i in range(n):
        # 原图
        plt.subplot(2, n, i+1)
        plt.imshow(x_denorm[i])
        plt.axis("off")
        # 重建图
        plt.subplot(2, 10, i+1+n)
        plt.imshow(recon_x_denorm[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def draw_prior_indices(real_indices, gen_indices, labels, num_embeddings=512,
                        class_map=None):
    """
    在 prior（PixelCNN）训练过程中，并排对比：
        上行 = VQ-VAE 真实编码的索引图（GT）
        下行 = PixelCNN 当前模型生成的索引图

    real_indices : (B, H, W) — 真实 VQ-VAE 编码的离散索引
    gen_indices  : (B, H, W) — PixelCNN 生成的离散索引
    labels       : (B,) — 对应的类别编号
    """
    B = real_indices.shape[0]
    n = min(B, 4)

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # 上行：真实 VQ-VAE 索引图
        im0 = axes[0, i].imshow(
            real_indices[i].cpu().numpy(), cmap='nipy_spectral',
            interpolation='nearest', vmin=0, vmax=num_embeddings - 1)
        lbl = labels[i].item()
        name = f" ({class_map[lbl]})" if class_map and lbl in class_map else ""
        axes[0, i].set_title(f"VQ-VAE GT  class {lbl}{name}", fontsize=10)
        axes[0, i].axis('off')

        # 下行：PixelCNN 生成索引图
        im1 = axes[1, i].imshow(
            gen_indices[i].cpu().numpy(), cmap='nipy_spectral',
            interpolation='nearest', vmin=0, vmax=num_embeddings - 1)
        axes[1, i].set_title(f"PixelCNN Gen  class {lbl}{name}", fontsize=10)
        axes[1, i].axis('off')

        # 统计重合度
        overlap = len(set(real_indices[i].flatten().tolist())
                      & set(gen_indices[i].flatten().tolist()))
        axes[0, i].text(0.5, -0.08,
                        f"real: {len(torch.unique(real_indices[i]))} unique  |  "
                        f"overlap: {overlap}",
                        transform=axes[0, i].transAxes,
                        ha='center', va='top', fontsize=8, color='gray')

    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Codebook Index", fontsize=10)

    plt.suptitle("Prior Training: VQ-VAE Encoded vs PixelCNN Generated",
                 fontsize=13, fontweight='bold')
    plt.subplots_adjust(left=0.02, right=0.92, top=0.88, bottom=0.08,
                        wspace=0.08, hspace=0.30)
    plt.show()


def train_prior(
    init_model,
    train_loader, valid_loader,
    epochs, print_epoc, lr, wd, device,
    num_embeddings=512,           # ← 码本总数，用于热力图色阶
    index_shape=(32, 32),          # ← generate() 的输出形状
    class_map=None                 # ← 可选 {class_idx: name}
):
    optimizer = optim.Adam(init_model.parameters(), lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    minn_val_loss = 1e20
    init_model = init_model.to(device)

    # ----- 用于可视化的固定样本（取第一个 batch 的前几张）-----
    vis_real = None   # (N, H, W)  真实 VQ-VAE 索引图
    vis_label = None  # (N,)       对应类别

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        samples = 0
        init_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            scores = init_model(x, y)
            loss = criterion(scores, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.shape[0]
            samples += x.shape[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 保存第一批样本用于后续可视化
            if vis_real is None:
                n_vis = min(x.shape[0], 4)
                vis_real = x[:n_vis].clone().cpu()
                vis_label = y[:n_vis].clone().cpu()

        epoch_loss = epoch_loss / samples
        train_loss_list.append(epoch_loss)

        # ---- 验证 ----
        init_model.eval()
        val_loss, val_samples = 0.0, 0
        with torch.no_grad():
            for x, y in tqdm(valid_loader, desc=f"Epoch {epoch}/{epochs} [valid]", leave=False):
                x, y = x.to(device).long(), y.to(device).long()
                scores = init_model(x, y)
                loss = criterion(scores, x)
                val_loss += loss.item() * x.shape[0]
                val_samples += x.shape[0]

        val_loss = val_loss / val_samples
        val_loss_list.append(val_loss)

        if epoch % print_epoc == 0:
            print(f"epoch: {epoch}\ntrain={epoch_loss:.4f}  val={val_loss:.4f}\n")
            # ---- 可视化：真实索引 vs 当前模型生成 ----
            if vis_real is not None:
                with torch.no_grad():
                    gen_idx = init_model.generate(
                        vis_label.to(device), index_shape,
                        batch_size=len(vis_label), temperature=1.0
                    )
                draw_prior_indices(vis_real, gen_idx.cpu(), vis_label,
                                   num_embeddings, class_map)

        if val_loss < minn_val_loss:
            minn_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in init_model.state_dict().items()}

    init_model.load_state_dict(best_state)
    return init_model, train_loss_list, val_loss_list


