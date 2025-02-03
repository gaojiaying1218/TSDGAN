import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from torchprofile import profile_macs
from tqdm import tqdm  # 用于显示进度条
import torchvision.models as models

from brisque import BRISQUE
from pytorch_fid import fid_score
from mpl_toolkits.mplot3d import Axes3D
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from libsvm.svmutil import *
import itertools
from scipy.interpolate import griddata

# 加载预训练的 VGG 模型
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_indices=[0, 5, 10]):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.ModuleList([vgg[i] for i in layer_indices])  # 提取指定层
        for param in self.parameters():
            param.requires_grad = False  # 冻结权重

    def forward(self, generated, target):
        loss = 0
        for layer in self.layers:
            generated = layer(generated)
            target = layer(target)
            loss += nn.functional.mse_loss(generated, target)
        return loss


class ResidualBlock_a(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_a, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        xa = x + self.conv_block(x)
        return xa  # 跳跃连接，将输入直接相加


class Generator_a(nn.Module):
    def __init__(self):
        super(Generator_a, self).__init__()

        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        # 多个残差块
        self.residuals = nn.Sequential(
            ResidualBlock_a(64),
            ResidualBlock_a(64),
            ResidualBlock_a(64),
            ResidualBlock_a(64),
            ResidualBlock_a(64)
        )

        # 输出卷积层
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 输出值限制在[-1, 1]
        )

    def forward(self, x):
        x_initial = self.initial(x)
        x_residual = self.residuals(x_initial)
        x_residual = x_residual.clone()
        x_initial = x_initial.clone()
        #x_final = self.output(x_residual.clone() + x_initial.clone())  # 将初始输入加到输出上，形成跳跃连接
        x_final = self.output(x_residual + x_initial)
        return x_final


class ResidualBlock_b(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock_b, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        xb = x + self.conv_block(x)
        return xb  # 跳跃连接，将输入直接相加


class Generator_b(nn.Module):
    def __init__(self):
        super(Generator_b, self).__init__()

        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False)
        )

        # 多个残差块
        self.residuals = nn.Sequential(
            ResidualBlock_b(64),
            ResidualBlock_b(64),
            ResidualBlock_b(64),
            ResidualBlock_b(64),
            ResidualBlock_b(64)
        )

        # 输出卷积层
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 输出值限制在[-1, 1]
        )

    def forward(self, x):
        x_initial = self.initial(x)
        x_residual = self.residuals(x_initial)
        x_residual = x_residual.clone()
        x_initial = x_initial.clone()
        #x_final = self.output(x_residual.clone() + x_initial.clone())  # 将初始输入加到输出上，形成跳跃连接
        x_final = self.output(x_residual + x_initial)
        return x_final



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Second convolutional layer with batch normalization
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Third convolutional layer with increased filters
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # Fourth convolutional layer with further increased filters
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Flatten the output for the fully connected layers
            nn.Flatten(),

            # First fully connected layer with dropout
            nn.Linear(512 * 8 * 8, 1024),  # Adjust according to the flattened size
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Second fully connected layer with dropout
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)



class OracleBoneDataset(Dataset):
    def __init__(self, low_quality_dir, high_quality_dir, skeletonized_dir, transform=None):
        self.low_quality_dir = low_quality_dir
        self.high_quality_dir = high_quality_dir
        self.skeletonized_dir = skeletonized_dir
        self.low_quality_images = sorted(os.listdir(low_quality_dir))
        self.high_quality_images = sorted(os.listdir(high_quality_dir))
        self.skeletonized_images = sorted(os.listdir(skeletonized_dir))
        self.transform = transform

    def __len__(self):
        return len(self.low_quality_images)

    def __getitem__(self, idx):
        # 加载低质量图片
        low_img_path = os.path.join(self.low_quality_dir, self.low_quality_images[idx])
        low_image = Image.open(low_img_path).convert("L")  # 假设图片是灰度图

        # 加载高质量图片
        high_img_path = os.path.join(self.high_quality_dir, self.high_quality_images[idx])
        high_image = Image.open(high_img_path).convert("L")  # 假设图片是灰度图

        # 加载骨架化图片
        skeletonized_img_path = os.path.join(self.skeletonized_dir, self.skeletonized_images[idx])
        skeletonized_image = Image.open(skeletonized_img_path).convert("L")  # 假设图片是灰度图

        # 应用变换（如归一化、转换为张量等）
        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)
            skeletonized_image = self.transform(skeletonized_image)

        return low_image, high_image, skeletonized_image  # 返回低质量图像、高质量图像和骨架化图像


# 定义骨架化函数
def skeletonize_image(image_path, output_path):
    # 读取灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 应用骨架化操作
    skeleton = cv2.ximgproc.thinning(image)
    # 保存骨架化图像
    cv2.imwrite(output_path, skeleton)


def save_generated_images(generator, epoch, low_quality_samples, output_dir):
    # 创建保存目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成图像
    with torch.no_grad():  # 不计算梯度
        generated_images = generator(low_quality_samples).cpu()  # 生成图像并移到CPU
        generated_images = (generated_images + 1) / 2  # 将图像归一化到[0, 1]
        low_quality_samples = (low_quality_samples + 1) / 2  # 将输入图像也归一化到[0, 1]

    # 保存图像
    grid_size = int(np.sqrt(len(generated_images)))  # 网格大小
    fig, axes = plt.subplots(grid_size, 2 * grid_size, figsize=(15, 10))  # 每行两列，左边是输入图像，右边是生成图像

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(generated_images):
                # 显示低质量输入图像（左边）
                axes[i, 2 * j].imshow(low_quality_samples[idx].cpu().permute(1, 2, 0).numpy(), cmap='gray')  # 转换为 NumPy 数组
                axes[i, 2 * j].axis('off')

                # 显示生成的图像（右边）
                axes[i, 2 * j + 1].imshow(generated_images[idx].cpu().permute(1, 2, 0).numpy(), cmap='gray')  # 转换为 NumPy 数组
                axes[i, 2 * j + 1].axis('off')

    plt.savefig(f"{output_dir}/generated_epoch_{epoch}.png")  # 保存图像
    plt.close(fig)  # 关闭图形以释放内存



# 指标计算函数
def compute_brisque(image_path):
    brisque = BRISQUE()
    return brisque.get_score(image_path)


def compute_fid(real_path, fake_path):
    # Define a transformation to resize and normalize images for InceptionV3
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize images to 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    ])

    # Get list of image paths in both real and fake directories
    real_image_paths = [os.path.join(real_path, fname) for fname in os.listdir(real_path) if
                        fname.endswith(('jpg', 'png', 'jpeg'))]
    fake_image_paths = [os.path.join(fake_path, fname) for fname in os.listdir(fake_path) if
                        fname.endswith(('jpg', 'png', 'jpeg'))]

    # Ensure both directories have the same number of images
    assert len(real_image_paths) == len(fake_image_paths), "Number of real and fake images must match"

    # Load and transform the images
    real_images = []
    fake_images = []

    for real_img_path, fake_img_path in zip(real_image_paths, fake_image_paths):
        # Open images
        real_image = Image.open(real_img_path).convert('RGB')
        fake_image = Image.open(fake_img_path).convert('RGB')

        # Apply transformation
        real_image = transform(real_image)
        fake_image = transform(fake_image)

        real_images.append(real_image)
        fake_images.append(fake_image)

    # Stack the images into batches
    real_images = torch.stack(real_images)
    fake_images = torch.stack(fake_images)

    # Save images temporarily to compute FID
    real_temp_path = 'real_temp'
    fake_temp_path = 'fake_temp'

    # Create temporary directories for storing resized images
    os.makedirs(real_temp_path, exist_ok=True)
    os.makedirs(fake_temp_path, exist_ok=True)

    for i, (real, fake) in enumerate(zip(real_images, fake_images)):
        # Save the images to the temporary directories
        real_save_path = os.path.join(real_temp_path, f"real_{i}.png")
        fake_save_path = os.path.join(fake_temp_path, f"fake_{i}.png")

        transforms.ToPILImage()(real).save(real_save_path)
        transforms.ToPILImage()(fake).save(fake_save_path)

    # Use the GPU-accelerated FID calculation method
    fid_value = fid_score.calculate_fid_given_paths(
        [real_temp_path, fake_temp_path],
        batch_size=4,
        device='cuda',
        dims=2048
    )

    # Clean up temporary directories
    for path in [real_temp_path, fake_temp_path]:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.rmdir(path)

    return fid_value



def compute_inception_score(images, batch_size=32, splits=10, device='cuda'):
    # Load the InceptionV3 model
    model = inception_v3(pretrained=True, transform_input=False).to(device).eval()

    # Define the transformation pipeline
    transform = Compose([
        Resize((299, 299)),  # Resize to match InceptionV3 input size
        ToTensor(),  # Convert PIL.Image to tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])

    # Transform and batch the images
    all_preds = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_tensors = []

        for img_path in batch_images:
            # Load and convert image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transform(img)
            batch_tensors.append(img)

        # Stack batch tensors and move to device
        batch_tensors = torch.stack(batch_tensors).to(device)

        # Predict using the model
        with torch.no_grad():
            batch_preds = torch.nn.functional.softmax(model(batch_tensors), dim=1)  # Apply softmax
        all_preds.append(batch_preds.cpu().numpy())

    # Combine all predictions
    preds = np.concatenate(all_preds, axis=0)

    # Compute the Inception Score
    epsilon = 1e-9  # Avoid numerical errors in log
    scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits: (i + 1) * len(preds) // splits]
        kl_div = part * (np.log(part + epsilon) - np.log(np.mean(part, axis=0) + epsilon))
        scores.append(np.exp(np.mean(np.sum(kl_div, axis=1))))

    return np.mean(scores), np.std(scores)



# 绘制实时折线图的函数
def update_plot(epoch, d_loss_a, g_loss_a, d_loss_b, g_loss_b):
    d_losses_a.append(d_loss_a.item())
    g_losses_a.append(g_loss_a.item())
    d_losses_b.append(d_loss_b.item())
    g_losses_b.append(g_loss_b.item())

    ax.clear()  # 清除当前图像内容
    ax.plot(range(1, epoch + 2), d_losses_a, label='D Loss A', color='red')
    ax.plot(range(1, epoch + 2), g_losses_a, label='G Loss A', color='blue')
    ax.plot(range(1, epoch + 2), d_losses_b, label='D Loss B', color='green')
    ax.plot(range(1, epoch + 2), g_losses_b, label='G Loss B', color='orange')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True)

    plt.pause(0.1)  # 暂停以更新图像内容
def load_models(generator_a, generator_b, load_path_a, load_path_b, device):
    generator_a.load_state_dict(torch.load(load_path_a, map_location=device))
    generator_b.load_state_dict(torch.load(load_path_b, map_location=device))
    generator_a.eval()  # 切换到评估模式
    generator_b.eval()

# 假设你的模型实例为 generator_a 和 generator_b
def save_models(generator_a, generator_b, save_path_a, save_path_b):
    torch.save(generator_a.state_dict(), save_path_a)
    torch.save(generator_b.state_dict(), save_path_b)



# 定义图片处理和保存函数
def process_and_save_images1(input_folder, output_folder_a, output_folder_b, generator_a, generator_b, device):
    # 创建输出文件夹
    os.makedirs(output_folder_a, exist_ok=True)
    os.makedirs(output_folder_b, exist_ok=True)

    # # 定义灰度图像的预处理

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 反归一化函数（将 [-1, 1] 转回 [0, 255]）
    def denormalize(tensor):
        tensor = tensor * 0.5 + 0.5  # 将 [-1, 1] 映射回 [0, 1]
        tensor = tensor * 255.0  # 将 [0, 1] 映射回 [0, 255]
        return tensor.clamp(0, 255).byte()

    # 遍历输入文件夹
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)

        # 读取图片
        img = Image.open(input_path).convert("RGB")  # 确保是 RGB 图像
        img_tensor = transform(img).unsqueeze(0).to(device)  # 添加 batch 维度

        # 使用生成模型 A
        with torch.no_grad():
            start_time = time.time()
            output_a = generator_a(img_tensor)
            end_time = time.time()
            # 计算时间
            time_per_image = end_time - start_time
            #print(f"Time to generate one image: {time_per_image:.6f} seconds")

        # 保存模型 A 的输出
        output_a_img = denormalize(output_a.squeeze(0).cpu())  # 移除 batch 维度
        output_a_img = transforms.ToPILImage()(output_a_img)
        output_a_path = os.path.join(output_folder_a, file_name)
        output_a_img.save(output_a_path)

        # 使用生成模型 B
        with torch.no_grad():
            output_b = generator_b(output_a)

        # 保存模型 B 的输出
        output_b_img = denormalize(output_b.squeeze(0).cpu())  # 移除 batch 维度
        output_b_img = transforms.ToPILImage()(output_b_img)
        output_b_path = os.path.join(output_folder_b, file_name)
        output_b_img.save(output_b_path)

    print(f"处理完成！图片已保存到：{output_folder_a} 和 {output_folder_b}")


# 输入和输出文件夹路径
def generator_loss(fake_images, real_images, discriminator, real_labels, lambda_gan, lambda_content, lambda_perceptual):
    """
       计算生成器的总损失函数。

       参数:
       - fake_images: 生成的假图像 (Tensor)
       - real_images: 真实图像 (Tensor)
       - discriminator: 判别器模型
       - real_labels: 真实标签 (Tensor)
       - lambda_gan: 对抗损失的权重 (float)
       - lambda_content: 内容损失的权重 (float)
       - lambda_perceptual: 感知损失的权重 (float)

       返回:
       - loss_g: 总损失 (Tensor)
       """
    # 对抗损失
    loss_gan = criterion_gan(discriminator(fake_images), real_labels)

    # 内容损失
    loss_content = criterion_content(fake_images, real_images)

    # 将灰度图扩展为 3 通道
    fake_images = fake_images.repeat(1, 3, 1, 1)  # (N, 1, H, W) -> (N, 3, H, W)
    real_images = real_images.repeat(1, 3, 1, 1)

    # 感知损失
    loss_perceptual = criterion_perceptual(fake_images, real_images)

    # 总损失
    loss_g = lambda_gan * loss_gan + lambda_content * loss_content + lambda_perceptual * loss_perceptual
    return loss_g



def train_gan(generator_a, generator_b, discriminator_a, discriminator_b,
              dataloader, dataset, num_epochs, device,
              optimizer_g_a, optimizer_g_b, optimizer_d_a, optimizer_d_b,
              criterion_gan, generator_loss,
              G_a_lambda_gan, G_a_lambda_content, G_a_lambda_perceptual,
              #G_b_lambda_gan, G_b_lambda_content, G_b_lambda_perceptual,
              noise_level=0.05, save_interval=5, output_dir_a='generated_G_skeleton_images', output_dir_b='generated_high_quality_images'):
    """
    训练GAN的函数，包括生成器A/B和判别器A/B。

    参数:
        generator_a, generator_b: PyTorch模型，分别表示生成骨架图像和高质量图像的生成器。
        discriminator_a, discriminator_b: PyTorch模型，分别表示骨架图像和高质量图像的判别器。
        dataloader: PyTorch DataLoader，用于加载训练数据。
        dataset: PyTorch Dataset，用于获取随机样本进行评估。
        num_epochs: 训练的总轮数。
        device: 设备（CPU或GPU）。
        optimizer_g_a, optimizer_g_b: 生成器A/B的优化器。
        optimizer_d_a, optimizer_d_b: 判别器A/B的优化器。
        criterion_gan: 损失函数，用于判别器。
        generator_loss: 自定义生成器损失函数。
        G_a_lambda_gan, G_a_lambda_content, G_a_lambda_perceptual: 生成器损失权重。
        noise_level: 添加到输入图像的噪声强度。
        save_interval: 保存生成图像的间隔（以epoch为单位）。
        output_dir_a, output_dir_b: 保存生成图像的目录。
    """
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        for low_images, high_images, skeleton_images in dataloader:
            # 将数据移动到设备
            low_images = low_images.to(device)
            high_images = high_images.to(device)
            skeleton_images = skeleton_images.to(device)

            # 为低质量图像添加噪声
            low_images_noisy = low_images + noise_level * torch.randn_like(low_images)

            # -----------------
            # 训练生成器A和判别器A
            # -----------------
            G_skeleton_images = generator_a(low_images_noisy)

            # 判别器A的标签
            real_labels = torch.empty(skeleton_images.size(0), 1, device=device).uniform_(0.9, 1.0)
            fake_labels = torch.empty(G_skeleton_images.size(0), 1, device=device).uniform_(0.0, 0.1)

            # 判别器A的损失
            d_loss_real_a = criterion_gan(discriminator_a(skeleton_images), real_labels)
            d_loss_fake_a = criterion_gan(discriminator_a(G_skeleton_images.detach()), fake_labels)
            d_loss_a = d_loss_real_a + d_loss_fake_a
            optimizer_d_a.zero_grad()
            d_loss_a.backward()
            optimizer_d_a.step()

            # 训练生成器A
            for _ in range(3):
                generated_skeleton = generator_a(low_images_noisy)
                g_loss_a = generator_loss(generated_skeleton, skeleton_images, discriminator_a, real_labels,
                                          G_a_lambda_gan, G_a_lambda_content, G_a_lambda_perceptual)
                optimizer_g_a.zero_grad()
                g_loss_a.backward()
                optimizer_g_a.step()

            # -----------------
            # 训练生成器B和判别器B
            # -----------------
            generated_skeleton_noisy = generated_skeleton + noise_level * torch.randn_like(generated_skeleton)
            generated_images = generator_b(generated_skeleton_noisy)

            real_labels_b = torch.empty(high_images.size(0), 1, device=device).uniform_(0.9, 1.0)
            fake_labels_b = torch.empty(generated_images.size(0), 1, device=device).uniform_(0.0, 0.1)

            # 判别器B的损失
            d_loss_real_b = criterion_gan(discriminator_b(high_images), real_labels_b)
            d_loss_fake_b = criterion_gan(discriminator_b(generated_images.detach()), fake_labels_b)
            d_loss_b = d_loss_real_b + d_loss_fake_b
            optimizer_d_b.zero_grad()
            d_loss_b.backward()
            optimizer_d_b.step()


            # 修正生成器 B 的训练部分
            for _ in range(3):
                # 重新计算 generated_skeleton_noisy，避免访问释放的图
                with torch.no_grad():  # 不需要计算梯度，仅用于生成输入
                    generated_skeleton_noisy = generator_a(low_images_noisy) + noise_level * torch.randn_like(
                        generated_skeleton)

                # 使用新的 generated_skeleton_noisy 训练生成器 B
                generated_images = generator_b(generated_skeleton_noisy)
                g_loss_b = generator_loss(generated_images, high_images, discriminator_b, real_labels_b,
                                          G_a_lambda_gan, G_a_lambda_content, G_a_lambda_perceptual)

                optimizer_g_b.zero_grad()
                g_loss_b.backward()  # 不需要 retain_graph=True
                optimizer_g_b.step()

        # 保存生成的样本图像
        if (epoch + 1) % save_interval == 0:
            indices = np.random.choice(len(dataset), size=4, replace=False)
            low_quality_samples = torch.stack([dataset[i][0] for i in indices]).to(device)

            save_generated_images(generator_a, epoch + 1, low_quality_samples, output_dir=output_dir_a)
            save_generated_images(generator_b, epoch + 1, generator_a(low_quality_samples), output_dir=output_dir_b)

        # 打印损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss A: {d_loss_a.item():.4f}, G Loss A: {g_loss_a.item():.4f}')
        print(f'Epoch [{epoch + 1}/{num_epochs}], D Loss B: {d_loss_b.item():.4f}, G Loss B: {g_loss_b.item():.4f}')

        # 实时更新折线图
        update_plot(epoch, d_loss_a, g_loss_a, d_loss_b, g_loss_b)

        lr_scheduler_d_a.step()
        lr_scheduler_g_a.step()
        lr_scheduler_d_b.step()
        lr_scheduler_g_b.step()
    return generator_a, generator_b


def save_and_load_models(generator_a, generator_b, save_path_a, save_path_b, load_path_a, load_path_b, device):
    """
    保存和加载生成器模型。

    参数:
        generator_a, generator_b: PyTorch模型，生成器A和B。
        save_path_a, save_path_b: 保存生成器A和B的路径。
        load_path_a, load_path_b: 加载生成器A和B的路径。
        device: 设备（CPU或GPU）。
    """
    # 保存模型
    torch.save(generator_a.state_dict(), save_path_a)
    torch.save(generator_b.state_dict(), save_path_b)
    print("模型已保存！")

    # 加载模型
    generator_a.load_state_dict(torch.load(load_path_a, map_location=device))
    generator_b.load_state_dict(torch.load(load_path_b, map_location=device))
    generator_a = generator_a.to(device)
    generator_b = generator_b.to(device)
    print("模型已加载！")
    return generator_a, generator_b


def compute_flops(generator, input_size, device):
    """
    计算模型的 FLOPs（浮点操作数）。

    参数:
        generator: PyTorch模型。
        input_size: 输入尺寸（例如 (1, 1, 256, 256)）。
        device: 设备（CPU或GPU）。

    返回:
        FLOPs 和 MACs。
    """
    dummy_input = torch.randn(*input_size).to(device)
    macs = profile_macs(generator, dummy_input)
    flops = macs * 2  # FLOPs 是 MACs 的 2 倍

    print(f"Multiply–Accumulate Operations (MACs): {macs}")
    print(f"FLOPs: {flops}")
    return macs, flops


def process_and_save_images(input_folder, output_folder_a, output_folder_b, generator_a, generator_b, device, transform=None):
    """
    使用生成器A和B处理输入图像并保存结果。

    参数:
        input_folder: 输入图像文件夹路径。
        output_folder_a: 生成器A输出图像的保存文件夹路径。
        output_folder_b: 生成器B输出图像的保存文件夹路径。
        generator_a, generator_b: PyTorch模型，生成器A和B。
        device: 设备（CPU或GPU）。
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder_a, exist_ok=True)
    os.makedirs(output_folder_b, exist_ok=True)

    # 处理每张图片
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        image = Image.open(input_path).convert('L')  # 假设单通道灰度图
        # 应用transform（如果定义了）
        if transform:
            image = transform(image)
        else:
            image = transforms.ToTensor()(image)  # 默认转换为张量

        image_tensor = image.unsqueeze(0).to(device)

        # 使用生成器A和B
        with torch.no_grad():
            generated_skeleton = generator_a(image_tensor)
            generated_image = generator_b(generated_skeleton)

        # 保存结果
        output_path_a = os.path.join(output_folder_a, filename)
        output_path_b = os.path.join(output_folder_b, filename)
        transforms.ToPILImage()(generated_skeleton.squeeze(0).cpu()).save(output_path_a)
        transforms.ToPILImage()(generated_image.squeeze(0).cpu()).save(output_path_b)

    print("图片处理完成，结果已保存！")


# 定义三维可视化函数
def plot_3d(lambda_combinations, fid_values, is_values, brisque_scores):
    fig = plt.figure(figsize=(18, 6))

    # 绘制 FID 的三维图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('FID')
    ax1.set_xlabel('G_a_lambda_content')
    ax1.set_ylabel('G_a_lambda_perceptual')
    ax1.set_zlabel('FID')
    ax1.scatter(lambda_combinations[:, 0], lambda_combinations[:, 1], fid_values, c='r')

    # 绘制 Inception Score 的三维图
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Inception Score')
    ax2.set_xlabel('G_a_lambda_content')
    ax2.set_ylabel('G_a_lambda_perceptual')
    ax2.set_zlabel('Inception Score')
    ax2.scatter(lambda_combinations[:, 0], lambda_combinations[:, 1], is_values, c='g')

    # 绘制 BRISQUE Score 的三维图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('BRISQUE Score')
    ax3.set_xlabel('G_a_lambda_content')
    ax3.set_ylabel('G_a_lambda_perceptual')
    ax3.set_zlabel('BRISQUE Score')
    ax3.scatter(lambda_combinations[:, 0], lambda_combinations[:, 1], brisque_scores, c='b')

    plt.savefig("example_plot.png")
    plt.show()


# 定义三维可视化函数（曲面图）
def plot_3d_surface(lambda_combinations, fid_values, is_values, brisque_scores):
    # 提取 G_a_lambda_content 和 G_a_lambda_perceptual
    x = lambda_combinations[:, 0]
    y = lambda_combinations[:, 1]

    # 定义网格
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # 对每个指标进行插值
    Z_fid = griddata((x, y), fid_values, (X, Y), method='cubic')
    Z_is = griddata((x, y), is_values, (X, Y), method='cubic')
    Z_brisque = griddata((x, y), brisque_scores, (X, Y), method='cubic')

    fig = plt.figure(figsize=(18, 6))

    # 绘制 FID 的曲面图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z_fid, cmap='viridis', edgecolor='none')
    ax1.set_title('FID')
    ax1.set_xlabel('G_a_lambda_content')
    ax1.set_ylabel('G_a_lambda_perceptual')
    ax1.set_zlabel('FID')

    # 绘制 Inception Score 的曲面图
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z_is, cmap='plasma', edgecolor='none')
    ax2.set_title('Inception Score')
    ax2.set_xlabel('G_a_lambda_content')
    ax2.set_ylabel('G_a_lambda_perceptual')
    ax2.set_zlabel('Inception Score')

    # 绘制 BRISQUE Score 的曲面图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, Z_brisque, cmap='coolwarm', edgecolor='none')
    ax3.set_title('BRISQUE Score')
    ax3.set_xlabel('G_a_lambda_content')
    ax3.set_ylabel('G_a_lambda_perceptual')
    ax3.set_zlabel('BRISQUE Score')

    plt.tight_layout()
    plt.savefig("example_surface_plot.png", dpi=300)
    plt.show()




if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转，角度范围为(-10°, 10°)
        transforms.RandomResizedCrop(128, scale=(0.9, 1.0)),  # 随机裁剪并调整大小
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变亮度、对比度、饱和度和色调
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 数据集路径
    high_quality_dir = r'D:\DataSet\H_L_RCD_Mix\High'  # 替换为你的低质量图片路径
    low_quality_dir = r'D:\DataSet\H_L_RCD_Mix\Low'  # 替换为你的高质量图片路径

    # 临时目录来保存骨架化后的高质量图片
    skeletonized_dir = r'D:\DataSet\H_L_RCD_Mix\High_bone'

    # 如果骨架化目录不存在或为空，则进行骨架化处理
    if not os.path.exists(skeletonized_dir) or len(os.listdir(skeletonized_dir)) == 0:
        os.makedirs(skeletonized_dir, exist_ok=True)

        # 对高质量图像进行骨架化并保存
        for filename in tqdm(os.listdir(high_quality_dir), desc="Skeletonizing images"):
            input_path = os.path.join(high_quality_dir, filename)
            output_path = os.path.join(skeletonized_dir, filename)

            # 检查目标文件是否已存在，避免重复处理
            if not os.path.exists(output_path):
                skeletonize_image(input_path, output_path)
        print("Skeletonization completed.")
    else:
        print("Skeletonized images already exist. Loading existing images.")

    # 设置 G_a_lambda_content 和 G_a_lambda_perceptual 的区间
    lambda_values = [x / 2 for x in range(5)]  # 从 0 到 5，间隔为 0.5，即 [0, 0.5, 1.0, ..., 5.0]

    # 生成所有超参数组合
    param_combinations = list(itertools.product(lambda_values, lambda_values))
    # 用于保存每个超参数组合的指标值
    fid_values = []
    is_values = []
    brisque_scores = []
    lambda_combinations = []

    # 在训练过程中的循环部分
    for G_a_lambda_content, G_a_lambda_perceptual in param_combinations:

        print(f"Training with G_a_lambda_content={G_a_lambda_content}, G_a_lambda_perceptual={G_a_lambda_perceptual}")

        # 定义生成器 A 和 B，以及判别器 A 和 B
        generator_a = Generator_a().to(device)
        discriminator_a = Discriminator().to(device)
        # summary(discriminator_a , input_size=(1, 1, 256, 256), device='cuda')
        generator_b = Generator_b().to(device)
        discriminator_b = Discriminator().to(device)

        # 损失函数和优化器
        criterion_gan = nn.BCELoss()
        criterion_content = nn.L1Loss()
        criterion_perceptual = VGGPerceptualLoss().to(device)


        initial_lr_g_a = 0.00025
        initial_lr_d_a = 0.0003
        initial_lr_g_b = 0.00025
        initial_lr_d_b = 0.0003

        # 定义优化器
        optimizer_g_a = torch.optim.Adam(generator_a.parameters(), lr=initial_lr_g_a)
        optimizer_d_a = torch.optim.Adam(discriminator_a.parameters(), lr=initial_lr_d_a)
        optimizer_g_b = torch.optim.Adam(generator_b.parameters(), lr=initial_lr_g_b)
        optimizer_d_b = torch.optim.Adam(discriminator_b.parameters(), lr=initial_lr_d_b)

        lr_scheduler_g_a = ExponentialLR(optimizer_g_a, gamma=0.995)
        lr_scheduler_d_a = ExponentialLR(optimizer_d_a, gamma=0.995)
        lr_scheduler_g_b = ExponentialLR(optimizer_g_b, gamma=0.995)
        lr_scheduler_d_b = ExponentialLR(optimizer_d_b, gamma=0.995)

        # 加载数据集
        dataset = OracleBoneDataset(low_quality_dir, high_quality_dir, skeletonized_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 初始化绘图数据
        d_losses_a = []
        g_losses_a = []
        d_losses_b = []
        g_losses_b = []


        # 创建图形
        plt.ion()  # 开启交互模式
        fig, ax = plt.subplots(figsize=(8, 6))

        torch.autograd.set_detect_anomaly(True)
        generator_a_trained, generator_b_trained = train_gan(generator_a, generator_b, discriminator_a, discriminator_b,
                  dataloader, dataset, num_epochs=55, device=device,
                  optimizer_g_a=optimizer_g_a, optimizer_g_b=optimizer_g_b,
                  optimizer_d_a=optimizer_d_a, optimizer_d_b=optimizer_d_b,
                  criterion_gan=criterion_gan, generator_loss=generator_loss,
                  G_a_lambda_gan=1, G_a_lambda_content=G_a_lambda_content, G_a_lambda_perceptual=G_a_lambda_perceptual,
                  #G_b_lambda_gan=0.1, G_b_lambda_content=1.0, G_b_lambda_perceptual=0.5,
                  noise_level=0.05, save_interval=5)
        plt.ioff()  # 关闭交互模式
        plt.show(block=False)  # 显示最终图像
        plt.close()


        # 计算 FLOPs
        compute_flops(generator_a_trained, input_size=(1, 1, 256, 256), device=device)
        compute_flops(generator_b_trained, input_size=(1, 1, 256, 256), device=device)

        # 处理并保存图像
        input_folder = r"D:\DataSet\H_L_RCD_Mix\Low"
        output_folder_a = r"D:\Code\RCD_Repair\Gs_outputing_LG_gan1_c5_p3"
        output_folder_b = r"D:\Code\RCD_Repair\Gt_outputing_LG_gan1_c5_p3"
        process_and_save_images1(input_folder, output_folder_a, output_folder_b, generator_a_trained, generator_b_trained, device)



        # 对每个超参数组合计算指标
        brisque_score = []
        for image_name in os.listdir(output_folder_b):
            image_path = os.path.join(output_folder_b, image_name)
            img = cv2.imread(image_path)
            brisque_score.append(compute_brisque(image_path))

        # FID 和 IS
        fid_value = compute_fid(high_quality_dir, output_folder_b)

        is_value, is_std = compute_inception_score(
            [os.path.join(output_folder_b, img) for img in os.listdir(output_folder_b)],
            device="cuda"
        )
        print(f"FID: {fid_value:.4f}")
        print(f"Inception Score: {is_value:.4f} ± {is_std:.4f}")

        # 将当前的超参数组合和指标值保存到列表中
        lambda_combinations.append([G_a_lambda_content, G_a_lambda_perceptual])
        fid_values.append(fid_value)
        is_values.append(is_value)
        brisque_scores.append(np.mean(brisque_score))  # 计算BRISQUE的平均值作为该超参数组合的得分

    # 转换为 NumPy 数组，便于绘图
    lambda_combinations = np.array(lambda_combinations)

    # # 绘制三维图
    # 调用函数
    plot_3d_surface(lambda_combinations, fid_values, is_values, brisque_scores)

    print("gjy")
