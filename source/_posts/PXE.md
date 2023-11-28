---
title: PXE 批量部署
date: 2023-11-27 18:21:00
excerpt: 南京大学软件所培训作业
author: dydy
tags:
- Shell
- Linux
- Qemu
- DHCP
- TFTP

categories:
- 南大
---

# PXE 批量部署

# 概述

PXE（Preboot Execution Environment，预启动执行环境）批量部署是一种通过网络对计算机系统进行安装和配置的技术。这种技术主要用于大规模部署操作系统，特别是在企业或教育机构等环境中，可以同时为多台计算机安装操作系统和软件。下面是 PXE 批量部署的主要步骤和特点：

1. **启动环境设置**：PXE 利用网络接口卡（NIC）的预引导功能，允许计算机在操作系统加载之前从网络启动。
2. **TFTP服务器**：PXE 客户端通过网络从 TFTP（Trivial File Transfer Protocol，简单文件传输协议）服务器下载所需的引导文件和操作系统映像。
3. **DHCP服务**：动态主机配置协议（DHCP）服务器为 PXE 客户端提供网络配置信息，如 IP 地址、子网掩码、默认网关和 TFTP 服务器地址。
4. **无需物理介质**：与传统的基于 CD/DVD 或 USB 驱动器的安装不同，PXE 批量部署不需要物理介质，大大简化了安装过程。
5. **自动化安装**：PXE 批量部署可以结合脚本和配置文件实现操作系统的自动化安装和配置，减少了手动干预的需求。
6. **适用于多种操作系统**：PXE 批量部署不限于特定的操作系统，可以用于安装多种不同的操作系统，包括 Windows、Linux 等。

PXE 批量部署特别适合需要在短时间内为大量计算机安装或重新安装操作系统的情况，大大提高了效率和一致性。

## KVM 与 TCG

### KVM 模式

KVM（Kernel-based Virtual Machine）是一种基于 Linux 内核的虚拟化技术。它允许你在 Linux 系统上运行多个带有自己操作系统的虚拟机，这些虚拟机可以是 Linux、Windows 或其他任何支持 x86 架构的操作系统。KVM 模式的特点包括：

1. **硬件辅助虚拟化**：KVM 利用现代处理器的硬件虚拟化支持（如 Intel VT 或 AMD-V）来提高性能。
2. **内核级运行**：作为 Linux 内核的一部分运行，提供高效的性能和优良的集成。
3. **支持多种客户机操作系统**：可以在 KVM 上运行各种操作系统。

### TCG 模式

TCG（Tiny Code Generator）是 QEMU（一种常用的虚拟化软件）使用的软件模拟模式。当硬件不支持虚拟化或者没有启用硬件虚拟化功能时，QEMU 会使用 TCG 来模拟 CPU，这意味着 CPU 指令是在软件层面上被模拟的。TCG 模式相比 KVM 模式性能较低，但它可以在不支持硬件虚拟化的环境下工作。

### 如何检查系统是否支持 KVM

要检查您的 Linux 系统是否支持 KVM，您可以使用以下命令：

1. **检查****CPU****是否支持硬件虚拟化**：
   
```bash
egrep -c '(vmx|svm)' /proc/cpuinfo
```

如果这个命令返回的数字大于0，那么您的CPU支持硬件虚拟化（vmx针对Intel处理器，svm针对AMD处理器）。

2. **检查****KVM****内核模块是否已加载**：

- 对于Intel处理器：

```bash
lsmod | grep kvm_intel
```

- 对于 AMD 处理器：
```bash
lsmod | grep kvm_amd
```

   如果这些命令返回了结果，表明相应的KVM模块已经被加载到内核中。

1. **检查是否安装了****KVM****工具：**

   可以通过检查`qemu-kvm`包是否安装来确定系统是否配置了KVM。

```bash
kvm-ok
```

如果您的系统安装了 `cpu-checker` 包，`kvm-ok` 命令会告诉您系统是否准备好运行 KVM 虚拟机。

## 启动

### 以命令行的方式启动虚拟机镜像

通过命令行启动虚拟机的镜像通常涉及到使用虚拟化工具，例如 QEMU 或 KVM。以下是使用 QEMU/KVM 通过命令行启动一个虚拟机镜像的基本步骤：

#### 使用 QEMU 启动虚拟机

1. **安装 QEMU**（如果尚未安装）：

在 Ubuntu 或 Debian 系统上，您可以使用以下命令安装 QEMU：

1. **启动虚拟机**：

   使用以下命令启动虚拟机镜像：

```bash
qemu-system-x86_64 -hda /path/to/your/vm/image.img
```

- `qemu-system-x86_64` 是 QEMU 用于 x86_64 架构的模拟器。
- `-hda` 指定硬盘镜像的位置，这里是您的虚拟机镜像的路径。

2. **其他选项**：

- 您可以添加 `-m` 参数来指定分配给虚拟机的内存量。例如，`-m 2048` 分配 2GB 内存。
- `-cdrom` 参数可以用来指定一个 ISO 镜像以模拟光驱。例如，`-cdrom /path/to/cdrom/image.iso`。


# 预备工具

## 准备 Ubuntu 镜像

从 [https://mirrors.nju.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.3-live-server-amd64.iso](https://mirrors.nju.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.3-live-server-amd64.iso) 下下载.iso 镜像，举个例子，放到 `~/Downloads` 目录下

## DHCP 服务器

1. 配置网络适配器

由于未来的虚拟机要在 `192.168.1.n` 的网络环境下进行通信，我们需要建造一个网桥

```bash
sudo brctl addbr br0
sudo ifup br0
sudo ip link set br0 up
```

1. 编辑 `/etc/network/interfaces` 文件

```bash
auto br0
iface br0 inet static
    address 192.168.1.1
    netmask 255.255.255.0
    network 192.168.1.0
    broadcast 192.168.1.255
```

2. 安装 `dhcp` 服务器

```bash
sudo apt install isc-dhcp-server
```

3. 配置 `dhcp` 服务器
   编辑 DHCP 服务器的配置文件 `/etc/dhcp/dhcpd.conf`，将配置文件加入到里头

```bash
subnet 192.168.1.0 netmask 255.255.255.0 {
    range 192.168.1.10 192.168.1.100;
    option domain-name-servers 8.8.8.8;
    option routers 192.168.1.1;
    option broadcast-address 192.168.1.255;
    default-lease-time 600;
    max-lease-time 7200;
}

allow booting;
allow bootp;

next-server 192.168.1.1;
filename "pxelinux.0";
```

这个配置指定了 DHCP 服务管理的子网、IP 地址范围、DNS 服务器地址、默认路由器（网关）地址和广播地址。另外 `filename` 定义了 `tftp` 服务器的相关配置，在下列文档中会提到

4. 指定网络接口

在 `/etc/default/isc-dhcp-server` 文件中指定 `DHCP` 服务应该监听的网络接口

```bash
INTERFACESv4="br0"
```

5. 使用 `systemctl` 启动 `dhcp` 服务

```bash
sudo systemctl start isc-dhcp-server.service
# 检查启动状态
sudo systemctl status isc-dhcp-server.service
```

## TFTP 服务器

1. 下载 `tftp` 服务器

```bash
sudo apt install tftpd-hpa
```

2. 配置 `tftp` 服务器

编辑 `/etc/default/tftp-hpa` 文件为

```bash
TFTP_USERNAME="tftp"
TFTP_DIRECTORY="/var/lib/tftpboot"
TFTP_ADDRESS="0.0.0.0:69"
TFTP_OPTIONS="--secure"
```

3. 设置 `tftp` 根目录

```bash
sudo mkdir /var/lib/tftpboot
sudo chown tftp:tftp /var/lib/tftpboot
sudo chmod -R 755 /var/lib/tftpboot
```

4. 设置 `qemu` 安装时的引导文件

编辑 `/var/lib/tftpboot/pxelinux.cfg/default` 文件

```bash
DEFAULT linux
LABEL linux
  SAY "Booting the Ubuntu 22.04 Installer..."
  KERNEL vmlinuz
  INITRD initrd
  APPEND root=/dev/ram0 ramdisk_size=1500000  ip=dhcp url=http://192.168.1.1/ubuntu-22.04.3-live-server-amd64.iso autoinstall ds=nocloud-net s=http://192.168.1.1/autoinstall/ cloud-config-url=http://192.168.1.1/autoinstall/user-data console=tty1 console=ttyS0 ---
```

- `DEFAULT linux`: 这指定了默认的标签或菜单项，当 PXE 引导时，如果没有用户交互，将自动选择这个菜单项。
- `LABEL linux`: 这定义了一个新的标签或菜单项，可以通过用户输入或默认设置被选择。
- `SAY "Booting the Ubuntu 22.04 Installer..."`: 这将在引导时在屏幕上打印一条消息，指示正在启动安装程序。
- `KERNEL vmlinuz`: 这指定了内核文件的位置，该文件是必须由 TFTP 服务器提供的，以便客户端下载并启动。
- `INITRD initrd`: 这指定了初始化 RAM 磁盘的位置，这也是由 TFTP 服务器提供，包含了内核在启动过程中所需的所有驱动程序和工具。
- `APPEND`: 这一行提供了内核启动时所需的额外参数：

  - `root=/dev/ram0`: 使用 RAM 磁盘作为根文件系统。
  - `ramdisk_size=1500000`: 设置 RAM 磁盘的大小（以千字节为单位）。
  - `ip=dhcp`: 通过 DHCP 获得 IP 地址。
  - `url=http://192.168.1.1/ubuntu-22.04.3-live-server-amd64.iso`: 指定 Ubuntu 安装 ISO 文件的位置，PXE 客户端将从这个 URL 下载 ISO 并进行安装。
  - `autoinstall`: 指定使用 autoinstall 方法进行无人值守安装。
  - `ds=nocloud-net s=http://192.168.1.1/autoinstall/`: 指定 autoinstall 配置文件的位置，它告诉安装程序如何进行无人值守安装。
  - `cloud-config-url=http://192.168.1.1/autoinstall/user-data`: 指定用户数据文件的 URL，它包含了安装过程中所需的用户和系统配置。
  - `console=tty1 console=ttyS0`: 指定控制台输出应该重定向到哪里，`tty1` 是第一个虚拟终端，`ttyS0` 是第一个串行端口。

> 以上 http 的内容将在 Apache 服务器的配置中实现

5. 放置资源文件

从刚才的 `.iso` 镜像中拿取响应的资源文件

```bash
sudo mkdir -p /mnt/ubuntu
sudo mount -o loop ~/Downloads/ubuntu-22.04.3-live-server-amd64.iso /mnt/ubuntu
sudo cp /mnt/casper/vmlinuz /var/lib/tftpboot/
sudo cp /mnt/casper/initrd /var/lib/tftpboot/
sudo umount /mnt/ubuntu
```

6. 启动 `tftp` 服务

配置完成后，启动 `tftp` 服务

```bash
sudo systemctl start tftp-hpa.service
sudo systemctl status tftp-hpa.service
```

## Apache 服务器

1. 安装 `Apache` 服务器

```bash
sudo apt install apache2
```

2. 配置 `Apache` 服务器

编辑 `/etc/apache2/apache2.conf` 文件，在最后一行指定 `Apache` 服务器监听的位置

```bash
serverName 192.168.1.1
```

3. 启动 `Apache` 服务

```bash
sudo systemctl start apache2
```

4. 开机自启动

```bash
sudo systemctl enable apache2
```

5. 检查 `Apache` 服务状态

```bash
sudo systemctl status apache2
```

6. 放置资源文件
   -  首先将.iso 镜像放到 `/var/www/html` 中
   -  放 `autoinstall` 目录

```bash
sudo mkdir -p /var/www/html/autoinstall
sudo touch /var/www/html/autoinstall/meta-data
sudo chmod -R 777 /var/www/html/autoinstall
```

   - 放置初始化脚本
```bash
sudo mkdir -p /var/www/html/init
sudo cp init/install.sh /var/www/html/init
sudo cp init/http_server.py /var/www/html/init
sudo cp init/http_server.service /var/www/html/init
```

- 放置ssh文件
```bash
sudo mkdir -p /var/www/html/ssh
sudo cp ~/.ssh/id_ed25519 /var/www/html/ssh
sudo cp ~/.ssh/id_ed25519.pub /var/www/html/ssh
```





# 构建流程

## 构建虚拟机镜像

```bash
bash build.sh 10
```

该过程会调用 `user_data.py` 自动生成 `user-data` 放到 `Apache` 服务器的目录下，然后生成 `vm-10` 目录

## 启动虚拟机

```bash
bash vm-10 run.sh
```

启动后安装相应工具，安装脚本已经预先安装在了 `/install.sh` 下

```bash
bash /install.sh
```

## 测试连通性

1. 通过主机请求虚拟机的端口获取虚拟机的机器编号
2. 测试虚拟机是否能够 ping 通主机
3. 测试主机是否能够 ping 通虚拟机
4. 测试虚拟机之间能否互相 ping 通

# 问题

## 网桥问题

- 问题描述

  - 要启动虚拟机的时候需要用到 `br0` 作为网桥，使用 `helper` 参数前需要先 allow 一下 br0
  - 启动脚本访问该文件的时候需要 `root` 权限
- 解决方案

  - 编辑 `/etc/qemu/bridge.conf` 为

```bash
allow br0
```
- 为`/usr/lib/qemu/qemu-bridge-helper`添加权限
```bash
sudo chmod u+s /usr/lib/qemu/qemu-bridge-helper
```

## MAC 地址重复

- 问题描述
  - 在启动多台虚拟机后，默认的多台虚拟机的 mac 地址会有重复
- 解决方案
  - 在启动时添加一个 mac 地址的参数，其中 14 表示虚拟机 20

```bash
mac=08:00:27:02:14:E7
```

## 虚拟机之间不能ping通

- 问题描述：
	- 虚拟机可以ping通主机
	- 主机可以ping通虚拟机
	- 虚拟机之间不能互相ping通
	- 虚拟机查看`arp -n`可以看到另一台虚拟机的IP地址
	- 虚拟机之间可以通过`arping`连接

- 解决方案
	- 这种情况下大概率是主机的防火墙问题，查看了一下`iptables`，发现里头的INPUT以及OUTPUT都是ACCEPT的，但是FORWARD选项是DROP的，这样表示，输入输出都没问题，但是禁止转发，通过下列命令进行修正

```bash
sudo iptables -P FORWARD ACCEPT
```
