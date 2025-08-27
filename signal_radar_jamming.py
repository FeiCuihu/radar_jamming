import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv
from datetime import datetime
import pywt
from scipy import io
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import re
import sqlite3
import uuid
from PIL import Image  # 用于调整图像尺寸

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 指定支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示为方块的问题


class SignalGenerator:
    """基础信号生成类，包含通用功能"""
    
    def __init__(self, sample_rate: float = 10e6, duration: float = 50e-6):
        """
        初始化信号生成器
        
        参数:
            sample_rate: 采样率 (Hz)
            duration: 信号总持续时间 (秒)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.t = np.arange(0, duration, 1/sample_rate)
        self.num_samples = len(self.t)
        self.silent_mode = False  # 新增静默模式标志

        # 默认输出目录
        self.output_dir = "signal_database_output"
        self._ensure_output_dir()
        
        # 计数器字典，用于跟踪不同信号类型的计数
        self._counters: Dict[str, int] = {}
        
        # 初始化数据库
        self.db_path = os.path.join(self.output_dir, "signal_database.db")
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.db_conn = sqlite3.connect(self.db_path)
        self.db_cursor = self.db_conn.cursor()
        
        # 创建信号记录表
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_records (
                id TEXT PRIMARY KEY,
                signal_type TEXT NOT NULL,
                generation_time TEXT NOT NULL,
                sample_rate REAL,
                duration REAL,
                num_samples INTEGER,
                parameters TEXT,
                snr_db REAL,
                sir_db REAL,
                file_path_base TEXT,
                comprehensive_plot_path TEXT,
                wavelet_plot_path TEXT,
                clean_wavelet_plot_path TEXT,
                clean_wavelet_grayscale_path TEXT
            )
        ''')
        self.db_conn.commit()
    
    def _store_record(self, record: Dict[str, Any]) -> str:
        """将信号记录存储到数据库"""
        signal_id = str(uuid.uuid4())
        
        # 准备参数JSON字符串
        params_str = json.dumps(record.get('params', {}))
        
        # 插入数据库记录
        self.db_cursor.execute('''
            INSERT INTO signal_records (
                id, signal_type, generation_time, 
                sample_rate, duration, num_samples, parameters,
                snr_db, sir_db, file_path_base,
                comprehensive_plot_path, wavelet_plot_path, clean_wavelet_plot_path, clean_wavelet_grayscale_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal_id,
            record.get('signal_type', ''),
            datetime.now().isoformat(),
            self.sample_rate,
            self.duration,
            self.num_samples,
            params_str,
            record.get('snr_db'),
            record.get('sir_db'),
            record.get('file_path_base', ''),
            record.get('comprehensive_plot_path', ''),
            record.get('wavelet_plot_path', ''),
            record.get('clean_wavelet_plot_path', ''),
            record.get('clean_wavelet_grayscale_path', '')
        ))
        self.db_conn.commit()
        return signal_id
    
    def _ensure_output_dir(self, path: Optional[str] = None) -> None:
        """确保输出目录存在"""
        if path:
            self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def set_silent_mode(self, silent: bool = True):
        """设置静默模式（不显示图像）"""
        self.silent_mode = silent

    def generate_lfm(self, f0: float, Kam: float, pulse_width: float, 
                    PRI: float, amplitude: float = 1.0) -> np.ndarray:
        """
        生成线性调频信号 (LFM)，支持多个脉冲
        
        参数:
            f0: 起始频率 (Hz)
            Kam: 调制斜率 (Hz/s)
            pulse_width: 脉冲宽度 (秒)
            PRI: 脉冲重复间隔 (秒)
            amplitude: 信号幅度
            
        返回:
            生成的LFM信号s
        """
        # 计算结束频率
        f1 = f0 + Kam * pulse_width
        
        # 初始化信号数组
        s = np.zeros(self.num_samples, dtype=np.complex128)
        
        # 计算脉冲数量
        num_pulses = int(np.floor(self.duration / PRI))
        
        # 生成每个脉冲
        for i in range(num_pulses):
            # 计算当前脉冲的开始时间
            pulse_start = i * PRI
            
            # 确定信号存在的时间段
            in_pulse = (self.t >= pulse_start) & (self.t < pulse_start + pulse_width)
            pulse_t = self.t[in_pulse] - pulse_start
            
            # 计算相位
            phase = 2 * np.pi * (f0 * pulse_t + 0.5 * Kam * pulse_t**2)
            s[in_pulse] = amplitude * np.exp(1j * phase)
        
        return s
    
    def add_noise(self, signal: np.ndarray, snr_db: Optional[float] = None) -> np.ndarray:
        """
        向信号添加高斯白噪声，控制信噪比(SNR)
        
        参数:
            signal: 输入信号
            snr_db: 信噪比 (dB)，如果为None则不添加噪声
            
        返回:
            添加噪声后的信号
        """
        if snr_db is None:
            return signal
            
        # 计算信号功率
        signal_power = np.mean(np.abs(signal)**2)
        
        # 计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # 生成复高斯噪声
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), self.num_samples)
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), self.num_samples)
        noise = noise_real + 1j * noise_imag
        
        return signal + noise
    
    def adjust_sir(self, signal: np.ndarray, interference: np.ndarray, sir_db: Optional[float] = None) -> np.ndarray:
        """
        调整干扰信号功率，控制信号与干扰比(SIR)
        
        参数:
            signal: 原始信号
            interference: 干扰信号
            sir_db: 信号与干扰比 (dB)，如果为None则不调整
            
        返回:
            调整后的干扰信号
        """
        if sir_db is None:
            return interference
            
        # 计算信号功率
        signal_power = np.mean(np.abs(signal)**2)
        
        # 计算干扰功率
        interference_power = np.mean(np.abs(interference)**2)
        
        if interference_power == 0:  # 避免除以零
            return interference
            
        # 计算需要的干扰功率
        target_interference_power = signal_power / (10 ** (sir_db / 10))
        
        # 调整干扰信号幅度
        scale_factor = np.sqrt(target_interference_power / interference_power)
        return interference * scale_factor

    def calculate_tf_concentration(self, coefficients: np.ndarray, method: str = 'renyi', alpha: float = 3.0) -> float:
        """
        计算小波变换时频表示的聚集度
        
        参数:
            coefficients: 小波变换系数矩阵
            method: 聚集度计算方法 ('renyi', 'shannon', 'variance')
            alpha: Rényi熵的参数（仅当method='renyi'时使用）
            
        返回:
            时频聚集度指标（值越小表示聚集度越高）
        """
        # 计算能量分布（使用系数的平方）
        energy = np.abs(coefficients) ** 2
        
        # 避免除以零和log(0)的问题
        energy = energy + 1e-12
        energy_norm = energy / np.sum(energy)
        
        if method == 'shannon':
            # 香农熵方法
            entropy = -np.sum(energy_norm * np.log2(energy_norm))
            return entropy
        
        elif method == 'renyi':
            # Rényi熵方法（对稀疏性更敏感）
            if alpha <= 1:
                alpha = 3.0  # 默认值
            
            if alpha == 1:
                # Rényi熵退化为香农熵
                entropy = -np.sum(energy_norm * np.log2(energy_norm))
            else:
                entropy = (1 / (1 - alpha)) * np.log2(np.sum(energy_norm ** alpha))
            return entropy
        
        elif method == 'variance':
            # 基于方差的方法
            # 计算时频平面的质心
            rows, cols = energy.shape
            x = np.arange(cols)
            y = np.arange(rows)
            
            # 计算质心坐标
            total_energy = np.sum(energy)
            centroid_x = np.sum(np.sum(energy, axis=0) * x) / total_energy
            centroid_y = np.sum(np.sum(energy, axis=1) * y) / total_energy
            
            # 计算二阶矩（方差）
            xx, yy = np.meshgrid(x, y)
            variance = np.sum(energy * ((xx - centroid_x) ** 2 + (yy - centroid_y) ** 2)) / total_energy
            
            return variance
        
        elif method == 'energy_ratio':
            # 基于能量比例的方法
            # 计算包含90%能量的最小区域大小
            sorted_energy = np.sort(energy.flatten())[::-1]  # 降序排列
            cumulative_energy = np.cumsum(sorted_energy)
            total_energy = cumulative_energy[-1]
            
            # 找到包含90%能量的最小元素数量
            idx_90 = np.where(cumulative_energy >= 0.9 * total_energy)[0][0]
            
            # 计算比例（越小表示能量越集中）
            ratio = idx_90 / energy.size
            return ratio
        
        else:
            raise ValueError(f"未知的聚集度计算方法: {method}")

    def save_signal(self, signal: np.ndarray, signal_id: str, params: Optional[Dict] = None, 
                   output_dir: Optional[str] = None, format_type: str = "npy", 
                   save_data: bool = False) -> str:
        """
        保存信号数据到文件
        
        参数:
            signal: 信号数据
            signal_id: 信号唯一ID
            params: 参数字典
            output_dir: 输出目录
            format_type: 保存格式 ('npy', 'mat', 'csv')
            save_data: 是否保存信号数据
        
        返回:
            保存的文件路径
        """
        # 确保输出目录存在
        if output_dir:
            self._ensure_output_dir(output_dir)
        
        # 使用UUID作为文件名
        file_path = os.path.join(self.output_dir, signal_id)
        
        # 保存信号数据（如果要求保存）
        if save_data:
            try:
                if format_type == "npy":
                    np.save(f"{file_path}.npy", signal)
                elif format_type == "mat":
                    io.savemat(f"{file_path}.mat", {'signal': signal, 't': self.t})
                elif format_type == "csv":
                    with open(f"{file_path}.csv", 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Time', 'Real', 'Imaginary'])
                        for i in range(min(10000, len(signal))):  # 限制行数
                            writer.writerow([self.t[i], np.real(signal[i]), np.imag(signal[i])])
                else:
                    raise ValueError(f"不支持的格式类型: {format_type}")
            except Exception as e:
                raise IOError(f"保存信号数据失败: {str(e)}")
        
        # 保存参数信息
        if params is not None:
            param_info = {
                'signal_id': signal_id,
                'signal_type': params.get('signal_type', 'unknown'),
                'sample_rate': self.sample_rate,
                'duration': self.duration,
                'num_samples': self.num_samples,
                'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'parameters': params
            }
            
            try:
                with open(f"{file_path}_params.json", 'w') as f:
                    json.dump(param_info, f, indent=4)
            except Exception as e:
                print(f"警告: 保存参数信息失败: {str(e)}")
        
        return file_path

    def plot_comprehensive(self, signal: np.ndarray, signal_id: str, signal_type: str,
                        f0: Optional[float] = None, f1: Optional[float] = None,
                        wavelet: str = 'cmor1.5-1.0') -> str:
        """
        综合绘制并保存信号的时域图、频域图和小波变换时频图，返回图像路径
        
        参数:
            signal: 信号数据
            signal_id: 信号唯一ID
            signal_type: 信号类型
            f0: 起始频率（用于小波变换）
            f1: 结束频率（用于小波变换）
            wavelet: 小波类型
            
        返回:
            图像保存路径
        """
        # 创建综合图
        fig = plt.figure(figsize=(15, 12))  # 增加高度以容纳新图表
        title = f"{signal_type}"
        
        # 时域图
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(self.t * 1e6, np.real(signal))
        ax1.set_title(f"{title} - 时域")
        ax1.set_xlabel("时间 (μs)")
        ax1.set_ylabel("幅度")
        ax1.grid(True)
        
        # 频域图
        ax2 = plt.subplot(2, 2, 2)
        freq = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/self.sample_rate))
        spectrum = np.fft.fftshift(np.abs(np.fft.fft(signal)))
        ax2.plot(freq / 1e6, 20*np.log10(spectrum + 1e-10))
        ax2.set_title(f"{title} - 频谱")
        ax2.set_xlabel("频率 (MHz)")
        ax2.set_ylabel("功率 (dB)")
        ax2.grid(True)
        ax2.set_ylim([-60, 80])
        
        # 小波变换时频图
        ax3 = plt.subplot(2, 2, 3)
        
        # 确定小波变换的频率范围
        if f0 is None or f1 is None:
            # 如果没有提供频率范围，则使用默认范围（0到采样率的一半）
            f0 = 0
            f1 = self.sample_rate / 2
        
        coefficients, frequencies = self.wavelet_transform(np.real(signal), f0, f1, wavelet)
        
        # 创建时频图
        im = ax3.imshow(np.abs(coefficients), 
                    extent=[0, self.duration*1e6, frequencies[-1]/1e6, frequencies[0]/1e6], 
                    aspect='auto', cmap='viridis')
        
        plt.colorbar(im, ax=ax3, label='幅度')
        ax3.set_title(f"{title} - 小波变换时频图")
        ax3.set_xlabel("时间 (μs)")
        ax3.set_ylabel("频率 (MHz)")
        
        # 计算时频聚集度
        try:
            tf_concentration = self.calculate_tf_concentration(coefficients, method='renyi')
            
            # 在时频图上添加文本框显示聚集度
            textstr = f'时频聚集度: {tf_concentration:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        except Exception as e:
            print(f"警告: 计算时频聚集度失败: {str(e)}")


        # 新增：计算并绘制瞬时频率标准差
        ax4 = plt.subplot(2, 2, 4)
        
        # 计算瞬时频率（使用复小波的相位信息）
        if wavelet.startswith('cmor'):  # 只有复小波才能计算瞬时频率
            try:
                # 提取相位
                phase = np.angle(coefficients)
                
                # 计算瞬时频率（对相位进行时间微分）
                instantaneous_freq = np.zeros_like(phase)
                sampling_interval = 1 / self.sample_rate
                
                for i in range(phase.shape[0]):
                    instantaneous_freq[i, :] = np.gradient(phase[i, :]) / (2 * np.pi * sampling_interval)
                
                # 使用幅度作为权重计算加权平均瞬时频率
                weights = np.abs(coefficients)
                weighted_instantaneous_freq = np.sum(instantaneous_freq * weights, axis=0) / np.sum(weights, axis=0)
                
                # 计算瞬时频率的标准差
                instantaneous_freq_std = np.std(weighted_instantaneous_freq)
                
                # 创建时间轴（微秒）
                time_axis = np.linspace(0, self.duration * 1e6, len(signal))
                
                # 绘制瞬时频率
                ax4.plot(time_axis, weighted_instantaneous_freq / 1e6, 'b-', label='瞬时频率')
                ax4.axhline(y=np.mean(weighted_instantaneous_freq) / 1e6, color='r', linestyle='--', 
                        label=f'均值: {np.mean(weighted_instantaneous_freq)/1e6:.2f} MHz')
                
                # 添加标准差带
                ax4.fill_between(time_axis,
                                (weighted_instantaneous_freq - instantaneous_freq_std) / 1e6,
                                (weighted_instantaneous_freq + instantaneous_freq_std) / 1e6,
                                alpha=0.3, color='gray',
                                label=f'±1 标准差: {instantaneous_freq_std/1e6:.2f} MHz')
                
                ax4.set_title(f"{title} - 瞬时频率 (标准差: {instantaneous_freq_std/1e6:.2f} MHz)")
                ax4.set_xlabel("时间 (μs)")
                ax4.set_ylabel("频率 (MHz)")
                ax4.legend()
                ax4.grid(True)
                
            except Exception as e:
                ax4.text(0.5, 0.5, f"无法计算瞬时频率: {str(e)}", 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title("瞬时频率计算失败")
        else:
            ax4.text(0.5, 0.5, "需要复小波(如cmor)来计算瞬时频率", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("不支持的小波类型")

        plt.tight_layout()
        
        # 生成图像路径
        image_path = os.path.join(self.output_dir, f"{signal_id}_{signal_type}_comprehensive.png")
        try:
            plt.savefig(image_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"警告: 保存综合图像失败: {str(e)}")
            image_path = ""
        
        if not self.silent_mode:
            plt.show()
        else:
            plt.close()  # 关闭图像以释放内存
        
        return image_path
    
    def wavelet_transform(self, signal: np.ndarray, f0: float, f1: float, wavelet: str = 'cmor1.5-1.0') -> Tuple[np.ndarray, np.ndarray]:
        """
        对信号进行小波变换
        
        参数:
            signal: 输入信号
            f0: 起始频率 (Hz)
            f1: 结束频率 (Hz)
            wavelet: 小波类型 (默认'cmor1.5-1.0')
        
        返回:
            小波系数和频率信息
        """
        # 计算合适的尺度范围，确保覆盖f0到f1的频率范围
        # 小波变换中，尺度与频率的关系：f = fc / (scale * dt)
        # 其中fc是小波的中心频率，dt是采样间隔
        fc = pywt.central_frequency(wavelet)  # 获取小波的中心频率
        dt = 1.0 / self.sample_rate  # 采样间隔
        
        # 计算对应f0和f1的尺度
        scale0 = fc / (f1 * dt)
        scale1 = fc / (f0 * dt)

        num_scales = 128  # 尺度数量，越多则频率分辨率越高，但计算量越大
        scales = np.logspace(np.log10(scale1), np.log10(scale0), num=num_scales, base=10)

        coefficients, frequencies = pywt.cwt(np.real(signal), scales, wavelet, sampling_period=1/self.sample_rate)
        
        return coefficients, frequencies
    
    def plot_wavelet(self, coefficients: np.ndarray, frequencies: np.ndarray, 
                    signal_id: str, signal_type: str, 
                    clean: bool = False, figsize: Tuple[int, int] = (10, 6),
                    target_size: Optional[Tuple[int, int]] = None,
                    output_format: str = "grayscale") -> Tuple[str, str]:
        """
        绘制小波变换结果，返回图像路径
        
        参数:
            coefficients: 小波系数
            frequencies: 频率信息
            signal_id: 信号唯一ID
            signal_type: 信号类型
            clean: 是否生成纯净图像（无坐标轴、标题等，适合CNN输入）
            figsize: 图像尺寸
            target_size: 目标图像尺寸 (宽度, 高度)，如果为None则使用figsize
            output_format: 输出格式 ('color', 'grayscale')
            
        返回:
            (彩色图像路径, 灰度图像路径) 元组
        """
        title = f"{signal_type}_{signal_id}"
        
        if clean:
            # 生成纯净图像，适合CNN输入
            plt.figure(figsize=figsize, frameon=False)
            
            # 根据输出格式选择colormap
            if output_format == "grayscale":
                cmap = 'gray'
            else:
                cmap = 'viridis'
                
            plt.imshow(np.abs(coefficients), aspect='auto', cmap=cmap)
            plt.axis('off')  # 关闭坐标轴
            suffix = "_wavelet_clean"
        else:
            # 生成完整图像，包含所有信息
            plt.figure(figsize=figsize)
            
            # 创建时频图
            plt.imshow(np.abs(coefficients), extent=[0, self.duration*1e6, frequencies[-1]/1e6, frequencies[0]/1e6], 
                      aspect='auto', cmap='viridis')
            
            plt.colorbar(label='幅度')
            plt.title(f"{title} - 小波变换")
            plt.xlabel("时间 (μs)")
            plt.ylabel("频率 (MHz)")
            suffix = "_wavelet"
        
        plt.tight_layout()
        
        # 生成图像路径
        color_image_path = os.path.join(self.output_dir, f"{signal_id}_{signal_type}{suffix}.png")
        grayscale_image_path = os.path.join(self.output_dir, f"{signal_id}_{signal_type}{suffix}_grayscale.png")
        
        try:
            if clean:
                if output_format == "grayscale":
                    plt.axis('off')
                    plt.savefig(color_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
                
                    # 如果需要调整尺寸
                    if target_size is not None:
                        self._resize_image(color_image_path, target_size)
                    
                    color_image_path = grayscale_image_path
                
                # # 如果需要生成灰度图
                # if output_format == "both" or output_format == "grayscale":
                #     self._convert_to_grayscale(color_image_path, grayscale_image_path, target_size)

                else:
                    plt.savefig(color_image_path, dpi=300, bbox_inches='tight', pad_inches=0)

                    if target_size is not None:
                        self._resize_image(color_image_path, target_size)
                    
                    # 如果需要生成灰度图
                    if output_format == "both":
                        self._convert_to_grayscale(color_image_path, grayscale_image_path, target_size)
            else:
                plt.savefig(color_image_path, dpi=300, bbox_inches='tight')
                
                # 如果需要生成灰度图
                if output_format == "both" or output_format == "grayscale":
                    self._convert_to_grayscale(color_image_path, grayscale_image_path)
        except Exception as e:
            print(f"警告: 保存小波图失败: {str(e)}")
            color_image_path = ""
            grayscale_image_path = ""
        
        if not self.silent_mode:
            plt.show()
        else:
            plt.close()  # 关闭图像以释放内存
            
        return color_image_path, grayscale_image_path
    
    '''
    def plot_wavelet(self, coefficients: np.ndarray, frequencies: np.ndarray, 
                 signal_id: str, signal_type: str, 
                 clean: bool = False, figsize: Tuple[int, int] = (10, 6),
                 target_size: Optional[Tuple[int, int]] = None,
                 output_format: str = "grayscale") -> Tuple[str, str]:
        """
        绘制小波变换结果，返回图像路径
        
        参数:
            coefficients: 小波系数
            frequencies: 频率信息
            signal_id: 信号唯一ID
            signal_type: 信号类型
            clean: 是否生成纯净图像（无坐标轴、标题等，适合CNN输入）
            figsize: 图像尺寸
            target_size: 目标图像尺寸 (宽度, 高度)，如果为None则使用figsize
            output_format: 输出格式 ('color', 'grayscale', 'both')
            
        返回:
            (彩色图像路径, 灰度图像路径) 元组
        """
        title = f"{signal_type}_{signal_id}"
        
        # 生成图像路径
        suffix = "_wavelet_clean" if clean else "_wavelet"
        color_image_path = os.path.join(self.output_dir, f"{signal_id}_{signal_type}{suffix}.png")
        grayscale_image_path = os.path.join(self.output_dir, f"{signal_id}_{signal_type}{suffix}_grayscale.png")
        
        try:
            if clean:
                # 生成纯净图像，适合CNN输入
                plt.figure(figsize=figsize, frameon=False)
                
                # 根据输出格式选择colormap
                cmap = 'gray' if output_format == "grayscale" else 'viridis'
                plt.imshow(np.abs(coefficients), aspect='auto', cmap=cmap)
                plt.axis('off')  # 关闭坐标轴
                
                # 保存图像
                if output_format == "grayscale":
                    plt.savefig(grayscale_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
                    color_image_path = ""  # 没有生成彩色图像
                else:
                    plt.savefig(color_image_path, dpi=300, bbox_inches='tight', pad_inches=0)
                    
                    # 如果需要生成灰度图
                    if output_format == "both":
                        self._convert_to_grayscale(color_image_path, grayscale_image_path)
                    
                    # 如果没有生成灰度图，确保路径为空
                    if output_format == "color":
                        grayscale_image_path = ""
                
                # 调整尺寸（如果指定了target_size）
                if target_size is not None:
                    if output_format != "grayscale" and color_image_path:
                        self._resize_image(color_image_path, target_size)
                    if output_format != "color" and grayscale_image_path:
                        self._resize_image(grayscale_image_path, target_size)
                        
            else:
                # 生成完整图像，包含所有信息
                plt.figure(figsize=figsize)
                
                # 创建时频图
                plt.imshow(np.abs(coefficients), extent=[0, self.duration*1e6, frequencies[-1]/1e6, frequencies[0]/1e6], 
                        aspect='auto', cmap='viridis')
                
                plt.colorbar(label='幅度')
                plt.title(f"{title} - 小波变换")
                plt.xlabel("时间 (μs)")
                plt.ylabel("频率 (MHz)")
                
                plt.tight_layout()
                plt.savefig(color_image_path, dpi=300, bbox_inches='tight')
                
                # 如果需要生成灰度图
                if output_format == "both" or output_format == "grayscale":
                    self._convert_to_grayscale(color_image_path, grayscale_image_path)
                else:
                    grayscale_image_path = ""
                    
        except Exception as e:
            print(f"警告: 保存小波图失败: {str(e)}")
            color_image_path = ""
            grayscale_image_path = ""
        
        finally:
            if not self.silent_mode:
                plt.show()
            else:
                plt.close()  # 关闭图像以释放内存
                
        return color_image_path, grayscale_image_path
    '''

    def _resize_image(self, image_path: str, target_size: Tuple[int, int]) -> None:
        """
        调整图像尺寸
        
        参数:
            image_path: 图像路径
            target_size: 目标尺寸 (宽度, 高度)
        """
        try:
            img = Image.open(image_path)
            img = img.resize(target_size, Image.LANCZOS)
            img.save(image_path)
        except Exception as e:
            print(f"警告: 调整图像尺寸失败: {str(e)}")
    
    def _convert_to_grayscale(self, color_image_path: str, grayscale_image_path: str, 
                             target_size: Optional[Tuple[int, int]] = None) -> None:
        """
        将彩色图像转换为灰度图像
        
        参数:
            color_image_path: 彩色图像路径
            grayscale_image_path: 灰度图像保存路径
            target_size: 目标尺寸 (宽度, 高度)，如果为None则保持原尺寸
        """
        try:
            img = Image.open(color_image_path).convert('L')  # 转换为灰度
            
            if img.mode != 'L':
                img = img.convert('L')
            # 如果需要调整尺寸
            if target_size is not None:
                img = img.resize(target_size, Image.LANCZOS)
                
            img.save(grayscale_image_path)
        except Exception as e:
            print(f"警告: 转换为灰度图像失败: {str(e)}")


class RadarJammingGenerator(SignalGenerator):
    """雷达干扰信号生成器，继承自SignalGenerator"""
    
    def __init__(self, sample_rate: float = 10e6, duration: float = 50e-6, 
                 radar_params: Optional[Dict[str, Any]] = None):
        """
        初始化雷达干扰信号生成器
        
        参数:
            sample_rate: 采样率 (Hz)
            duration: 信号总持续时间 (秒)
            radar_params: 雷达信号参数字典，包含f0, Kam, pulse_width, PRI, amplitude
        """
        super().__init__(sample_rate, duration)
        
        # 设置默认雷达参数
        if radar_params is None:
            radar_params = {
                'f0': 1e6,
                'Kam': (5e6 - 1e6) / 30e-6,  # 调制斜率 (Hz/s)
                'pulse_width': 30e-6,
                'PRI': 100e-6,
                'amplitude': 1.0
            }
        # 用于跟踪欺骗参数的状态
        self.range_deception_counter = 0
        self.velocity_deception_counter = 1
        self.last_delay_time = 0
        self.last_doppler_shift = 0

        # 生成默认的雷达信号 (LFM)
        self.radar_signal = self.generate_lfm(**radar_params)
        self.radar_params = radar_params
        
        # 计算结束频率
        self.radar_params['f1'] = radar_params['f0'] + radar_params['Kam'] * radar_params['pulse_width']

        self.DEFAULT_CONFIG = {
            'save_format': 'npy',
            'generate_plots': False,
            'analyze_wavelet': True,
            'snr_db': None,
            'sir_db': None,
            'save_data': False,
            'clean_wavelet': True,
            'comprehensive_plot': True,
            'add_radar_signal': True,
            'wavelet_target_size': (256, 256),
            'wavelet_output_format': 'grayscale'
        }
    
    def update_radar_signal(self, radar_params: Dict[str, Any]) -> None:
        """
        更新雷达信号
        
        参数:
            radar_params: 雷达信号参数字典，包含f0, Kam, pulse_width, PRI, amplitude
        """
        self.radar_signal = self.generate_lfm(**radar_params)
        self.radar_params = radar_params
        # 计算结束频率
        self.radar_params['f1'] = radar_params['f0'] + radar_params['Kam'] * radar_params['pulse_width']
    
    def noise_fm_jamming(self, center_freq: float, freq_deviation: float, jamming_power: float = 1, 
                         add_radar_signal: bool = True) -> np.ndarray:
        """
        生成噪声调频干扰 (NFM)，并叠加到雷达信号上
        
        参数:
            center_freq: 中心频率 (Hz)
            freq_deviation: 频率偏差 (Hz)
            jamming_power: 相对干扰功率
            add_radar_signal: 是否将干扰叠加到雷达信号上
            
        返回:
            噪声调频干扰信号（可能包含雷达信号）
        """
        # 生成零均值高斯噪声
        noise = np.random.normal(0, 1, self.num_samples)
        
        # 计算瞬时频率
        instant_freq = center_freq + freq_deviation * noise
        
        # 计算相位变化
        phase = 2 * np.pi * np.cumsum(instant_freq) / self.sample_rate
        
        # 生成复数干扰信号
        amplitude = np.sqrt(jamming_power)
        jamming_signal = amplitude * np.exp(1j * phase)
        
        # 将干扰信号叠加到雷达信号上
        if add_radar_signal:
            return self.radar_signal + jamming_signal
        else:
            return jamming_signal
    
    def noise_am_jamming(self, center_freq: float, bandwidth: float, jamming_power: float = 1,
                         add_radar_signal: bool = True) -> np.ndarray:
        """
        生成噪声调幅干扰 (NAM)，并叠加到雷达信号上
        
        参数:
            center_freq: 中心频率 (Hz)
            bandwidth: 带宽 (Hz)
            jamming_power: 相对干扰功率
            add_radar_signal: 是否将干扰叠加到雷达信号上
            
        返回:
            噪声调幅干扰信号（可能包含雷达信号）
        """
        # 生成噪声包络
        noise_envelope = np.random.uniform(0, 1, self.num_samples)
        
        # 生成载波信号
        carrier = np.cos(2 * np.pi * center_freq * self.t)
        
        # 调制载波
        amplitude = np.sqrt(jamming_power)
        jamming_signal = amplitude * noise_envelope * carrier
        
        # 将干扰信号叠加到雷达信号上
        if add_radar_signal:
            return self.radar_signal + jamming_signal
        else:
            return jamming_signal
    
    def dense_false_targets(self, num_targets: int, jamming_power: float = 1,
                            add_radar_signal: bool = True) -> np.ndarray:
        """
        生成密集假目标干扰 (DFT)，并叠加到雷达信号上
        
        参数:
            num_targets: 假目标数量
            jamming_power: 相对干扰功率
            add_radar_signal: 是否将干扰叠加到雷达信号上
            
        返回:
            密集假目标干扰信号（可能包含雷达信号）
        """
        # 创建信号副本
        s = np.zeros_like(self.radar_signal, dtype=np.complex128)
        
        # 生成多个假目标
        for _ in range(num_targets):
            # 随机延迟和幅度
            delay = np.random.randint(0, self.num_samples // 2)
            amplitude = np.sqrt(jamming_power) * np.random.uniform(0.2, 1.0)
            
            # 添加延迟的雷达信号副本
            delay_sig = np.roll(self.radar_signal, delay) * amplitude
            s += delay_sig
        
        # 将干扰信号叠加到雷达信号上
        if add_radar_signal:
            return self.radar_signal + s
        else:
            return s
    
    def range_deception_jamming(self, delay_time: float, jamming_power: float = 1,
                               add_radar_signal: bool = True) -> np.ndarray:
        """
        生成距离欺骗干扰 (RD)，并叠加到雷达信号上

        参数:
            delay_time: 延迟时间 (秒)，模拟假目标的距离延迟
            jamming_power: 相对干扰功率（相对于雷达信号的倍数）
            add_radar_signal: 是否将干扰叠加到雷达信号上
            
        返回:
            距离欺骗干扰信号（可能包含雷达信号）
        """
        # 获取雷达信号参数
        pulse_width = self.radar_params['pulse_width']
        PRI = self.radar_params['PRI']
        
        # 计算脉冲数量
        num_pulses = int(np.floor(self.duration / PRI))
        
        # 初始化干扰信号
        jamming_signal = np.zeros_like(self.radar_signal, dtype=np.complex128)
        
        # 计算延迟采样点数
        delay_samples = int(delay_time * self.sample_rate)
        
        # 确保延迟时间合理（不超过脉冲宽度）
        if delay_samples >= int(pulse_width * self.sample_rate):
            delay_samples = int(0.5 * pulse_width * self.sample_rate)
            print(f"警告: 延迟时间过长，已调整为脉冲宽度的一半: {delay_samples/self.sample_rate*1e6:.2f} μs")
        
        # 对每个脉冲生成干扰
        for i in range(num_pulses):
            # 计算当前脉冲的开始时间
            pulse_start = i * PRI
            
            # 干扰开始时间（雷达脉冲开始时间 + 延迟时间）
            jamming_start = pulse_start + delay_time
            jamming_end = min(jamming_start + pulse_width, self.duration)
            
            # 确保干扰时间段有效
            if jamming_start >= self.duration or jamming_end <= jamming_start:
                continue
            
            # 确定干扰时间段
            jamming_start_idx = int(jamming_start * self.sample_rate)
            jamming_end_idx = int(jamming_end * self.sample_rate)
            
            # 确定对应的原始雷达信号时间段
            radar_start_idx = int(pulse_start * self.sample_rate)
            radar_end_idx = int((pulse_start + pulse_width) * self.sample_rate)
            
            # 计算需要复制的长度
            copy_length = min(
                jamming_end_idx - jamming_start_idx,  # 干扰段的长度
                radar_end_idx - radar_start_idx       # 雷达段的长度
            )
            
            if copy_length <= 0:
                continue
            
            # 提取雷达信号部分
            radar_segment = self.radar_signal[radar_start_idx:radar_start_idx + copy_length]
            
            # 应用干扰功率调整
            jamming_amplitude = np.sqrt(jamming_power)
            jamming_signal[jamming_start_idx:jamming_start_idx + copy_length] = jamming_amplitude * radar_segment
        
        # 将干扰信号叠加到雷达信号上
        if add_radar_signal:
            return self.radar_signal + jamming_signal
        else:
            return jamming_signal
    
    def velocity_deception_jamming(self, doppler_shift: float, jamming_power: float = 1,
                              add_radar_signal: bool = True) -> np.ndarray:
        """
        生成速度欺骗干扰 (VD)，并叠加到雷达信号上
        
        参数:
            doppler_shift: 多普勒频移 (Hz)，正值表示目标远离，负值表示目标靠近
            jamming_power: 相对干扰功率（相对于雷达信号的倍数）
            add_radar_signal: 是否将干扰叠加到雷达信号上
            
        返回:
            速度欺骗干扰信号（可能包含雷达信号）
        """
        # 获取雷达信号参数
        pulse_width = self.radar_params['pulse_width']
        PRI = self.radar_params['PRI']
        
        # 计算脉冲数量
        num_pulses = int(np.floor(self.duration / PRI))
        
        # 初始化干扰信号
        jamming_signal = np.zeros_like(self.radar_signal, dtype=np.complex128)
        
        # 对每个脉冲生成干扰
        for i in range(num_pulses):
            # 计算当前脉冲的开始时间
            pulse_start = i * PRI
            
            # 干扰开始时间（从雷达信号开始后延迟一段时间，模拟处理时间）
            processing_delay = 2e-6  # 处理延迟时间（秒）
            jamming_start = pulse_start + processing_delay
            jamming_end = min(pulse_start + pulse_width, self.duration)
            
            if jamming_start >= jamming_end:
                continue
                
            # 确定干扰时间段
            in_jamming = (self.t >= jamming_start) & (self.t < jamming_end)
            jamming_t = self.t[in_jamming] - jamming_start
            
            # 提取对应的雷达信号部分
            radar_segment = self.radar_signal[in_jamming]
            
            # 应用多普勒频移 - 使用相位旋转而不是重新生成信号
            # 多普勒效应会导致相位变化: φ = 2π * f_d * t
            phase_shift = 2 * np.pi * doppler_shift * jamming_t
            doppler_shifted = radar_segment * np.exp(1j * phase_shift)
            
            # 应用干扰功率调整
            jamming_amplitude = np.sqrt(jamming_power)
            jamming_signal[in_jamming] = jamming_amplitude * doppler_shifted
        
        # 将干扰信号叠加到雷达信号上
        if add_radar_signal:
            return self.radar_signal + jamming_signal
        else:
            return jamming_signal
    
    def composite_jamming(self, jamming_components: List[Dict[str, Any]],
                         add_radar_signal: bool = True) -> np.ndarray:
        """
        生成复合干扰信号，并叠加到雷达信号上
        
        参数:
            jamming_components: 干扰组件列表，每个元素为字典:
                {
                    'type': 干扰类型,
                    'params': 干扰参数
                }
            add_radar_signal: 是否将干扰叠加到雷达信号上
                
        返回:
            复合干扰信号（可能包含雷达信号）
        """
        s = np.zeros_like(self.radar_signal)
        
        # 处理每个干扰组件
        for component in jamming_components:
            jam_type = component['type']
            params = component['params']
            
            if jam_type == 'noise_fm':
                # 临时生成干扰信号（不叠加雷达信号）
                jamming_signal = self.noise_fm_jamming(
                    params['center_freq'], 
                    params['freq_deviation'], 
                    params['jamming_power'],
                    add_radar_signal=False
                )
                s += jamming_signal
                
            elif jam_type == 'noise_am':
                # 临时生成干扰信号（不叠加雷达信号）
                jamming_signal = self.noise_am_jamming(
                    params['center_freq'], 
                    params['bandwidth'], 
                    params['jamming_power'],
                    add_radar_signal=False
                )
                s += jamming_signal
                
            elif jam_type == 'dense_false':
                # 临时生成干扰信号（不叠加雷达信号）
                jamming_signal = self.dense_false_targets(
                    params['num_targets'], 
                    params['jamming_power'],
                    add_radar_signal=False
                )
                s += jamming_signal
                
            elif jam_type == 'range_deception':
                # 临时生成干扰信号（不叠加雷达信号）
                jamming_signal = self.range_deception_jamming(
                    params['delay_time'], 
                    params['jamming_power'],
                    add_radar_signal=False
                )
                s += jamming_signal
                
            elif jam_type == 'velocity_deception':
                # 临时生成干扰信号（不叠加雷达信号）
                jamming_signal = self.velocity_deception_jamming(
                    params['doppler_shift'], 
                    params['jamming_power'],
                    add_radar_signal=False
                )
                s += jamming_signal
            else:
                raise ValueError(f"未知干扰类型: {jam_type}")
        
        # 将干扰信号叠加到雷达信号上
        if add_radar_signal:
            return self.radar_signal + s
        else:
            return s
    
    def generate_signal(self, signal_type: str, params: Dict[str, Any], 
                       snr_db: Optional[float] = None, sir_db: Optional[float] = None,
                       add_radar_signal: bool = True,radar_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        仅生成信号，不进行分析和保存
        
        参数:
            signal_type: 信号类型 ('radar', 'noise_fm', 'noise_am', 'dense_false', 
                         'range_deception', 'velocity_deception', 'composite')
            params: 信号参数
            snr_db: 信噪比 (dB)，如果为None则不添加噪声
            sir_db: 信号与干扰比 (dB)，仅对干扰信号有效，如果为None则不调整
            add_radar_signal: 是否将干扰叠加到雷达信号上
        
        返回:
            生成的信号signal
        """

        print(f"生成信号类型: {signal_type}")
        print(f"参数: {params}")

        if radar_params is not None:
            original_params = getattr(self, 'radar_params', None)

        # 生成信号
        if signal_type == 'radar':
            signal = self.generate_lfm(**params)
            signal = self.add_noise(signal, snr_db)
            
        elif signal_type == 'noise_fm':
            # 先生成干扰信号（包含雷达信号）
            signal = self.noise_fm_jamming(
                params['center_freq'], 
                params['freq_deviation'], 
                params['jamming_power'],
                add_radar_signal=add_radar_signal
            )
            
            # 调整干扰功率以达到目标SIR
            if sir_db is not None and add_radar_signal:
                # 提取干扰部分
                interference = signal - self.radar_signal
                # 调整干扰功率
                adjusted_interference = self.adjust_sir(self.radar_signal, interference, sir_db)
                # 重新组合信号
                signal = self.radar_signal + adjusted_interference
            
            # 添加噪声
            signal = self.add_noise(signal, snr_db)
            
        elif signal_type == 'noise_am':
            # 先生成干扰信号（包含雷达信号）
            signal = self.noise_am_jamming(
                params['center_freq'], 
                params['bandwidth'], 
                params['jamming_power'],
                add_radar_signal=add_radar_signal
            )
            
            # 调整干扰功率以达到目标SIR
            if sir_db is not None and add_radar_signal:
                # 提取干扰部分
                interference = signal - self.radar_signal
                # 调整干扰功率
                adjusted_interference = self.adjust_sir(self.radar_signal, interference, sir_db)
                # 重新组合信号
                signal = self.radar_signal + adjusted_interference
            
            # 添加噪声
            signal = self.add_noise(signal, snr_db)
            
        elif signal_type == 'dense_false':
            # 先生成干扰信号（包含雷达信号）
            signal = self.dense_false_targets(
                params['num_targets'], 
                params['jamming_power'],
                add_radar_signal=add_radar_signal
            )
            
            # 调整干扰功率以达到目标SIR
            if sir_db is not None and add_radar_signal:
                # 提取干扰部分
                interference = signal - self.radar_signal
                # 调整干扰功率
                adjusted_interference = self.adjust_sir(self.radar_signal, interference, sir_db)
                # 重新组合信号
                signal = self.radar_signal + adjusted_interference
            
            # 添加噪声
            signal = self.add_noise(signal, snr_db)
            
        elif signal_type == 'range_deception':
            # 先生成干扰信号（包含雷达信号）
            signal = self.range_deception_jamming(
                params['delay_time'], 
                params['jamming_power'],
                add_radar_signal=add_radar_signal
            )
            
            # 调整干扰功率以达到目标SIR
            if sir_db is not None and add_radar_signal:
                # 提取干扰部分
                interference = signal - self.radar_signal
                # 调整干扰功率
                adjusted_interference = self.adjust_sir(self.radar_signal, interference, sir_db)
                # 重新组合信号
                signal = self.radar_signal + adjusted_interference
            
            # 添加噪声
            signal = self.add_noise(signal, snr_db)
            
        elif signal_type == 'velocity_deception':
            # 先生成干扰信号（包含雷达信号）
            signal = self.velocity_deception_jamming(
                params['doppler_shift'], 
                params['jamming_power'],
                add_radar_signal=add_radar_signal
            )
            
            # 调整干扰功率以达到目标SIR
            if sir_db is not None and add_radar_signal:
                # 提取干扰部分
                interference = signal - self.radar_signal
                # 调整干扰功率
                adjusted_interference = self.adjust_sir(self.radar_signal, interference, sir_db)
                # 重新组合信号
                signal = self.radar_signal + adjusted_interference
            
            # 添加噪声
            signal = self.add_noise(signal, snr_db)
            
        elif signal_type == 'composite':
            # 先生成干扰信号（包含雷达信号）
            signal = self.composite_jamming(
                params['components'],
                add_radar_signal=add_radar_signal
            )
            
            # 调整干扰功率以达到目标SIR
            if sir_db is not None and add_radar_signal:
                # 提取干扰部分
                interference = signal - self.radar_signal
                # 调整干扰功率
                adjusted_interference = self.adjust_sir(self.radar_signal, interference, sir_db)
                # 重新组合信号
                signal = self.radar_signal + adjusted_interference
            
            # 添加噪声
            signal = self.add_noise(signal, snr_db)
            
        else:
            raise ValueError(f"未知信号类型: {signal_type}")
        
        if np.all(signal == 0):
            print(f"警告: {signal_type}信号全为零!")

        return signal
    
    def generate_and_analyze(self, signal_type: str, params: Dict[str, Any], output_dir: Optional[str] = None, 
                             save_format: str = "npy", analyze_wavelet: bool = True,
                             snr_db: Optional[float] = None, sir_db: Optional[float] = None,
                             save_data: bool = False, clean_wavelet: bool = False,
                             comprehensive_plot: bool = True, show_plots: bool = False,
                             add_radar_signal: bool = True,
                             wavelet_target_size: Optional[Tuple[int, int]] = None,
                             wavelet_output_format: str = "grayscale") -> Dict[str, Any]:
        """
        生成信号并进行完整分析（包括小波变换）
        
        参数:
            signal_type: 信号类型 ('radar', 'noise_fm', 'noise_am', 'dense_false', 
                         'range_deception', 'velocity_deception', 'composite')
            params: 信号参数
            output_dir: 输出目录
            save_format: 保存格式
            generate_plots: 是否生成图像
            analyze_wavelet: 是否进行小波分析
            snr_db: 信噪比 (dB)，如果为None则不添加噪声
            sir_db: 信号与干扰比 (dB)，仅对干扰信号有效，如果为None则不调整
            save_data: 是否保存信号数据
            clean_wavelet: 是否生成纯净的小波图像（无坐标轴、标题等，适合CNN输入）
            comprehensive_plot: 是否生成综合图像（时域、频域和时频域）
            show_plots: 是否显示图像
            add_radar_signal: 是否将干扰叠加到雷达信号上
            wavelet_target_size: 小波图像目标尺寸 (宽度, 高度)
            wavelet_output_format: 小波图像输出格式 ('color', 'grayscale', 'both')
            
        返回:
            结果字典，包含信号和文件路径信息
        """
        # 生成信号
        signal = self.generate_signal(signal_type, params, snr_db, sir_db, add_radar_signal)
        
        # 生成唯一ID
        signal_id = str(uuid.uuid4())
        
        # 保存信号（如果要求保存）
        base_path = self.save_signal(
            signal, 
            signal_id=signal_id, 
            params={**params, 'signal_type': signal_type},
            output_dir=output_dir,
            format_type=save_format,
            save_data=save_data
        )
        
        result = {
            'signal': signal,
            'signal_type': signal_type,
            'signal_id': signal_id,
            'file_path_base': base_path,
            'params': params
        }
        
        # 添加SNR和SIR信息到结果
        if snr_db is not None:
            result['snr_db'] = snr_db
        if sir_db is not None:
            result['sir_db'] = sir_db
        
        # 确定小波变换的频率范围
        if signal_type == 'radar':
            f0_wavelet = params['f0']
            # 计算结束频率
            f1_wavelet = params['f0'] + params['Kam'] * params['pulse_width']
        else:
            # 对于干扰信号，使用雷达信号的频率范围
            f0_wavelet = self.radar_params['f0']
            f1_wavelet = self.sample_rate / 2
        
        # 生成图像
        comprehensive_plot_path = ""
        wavelet_plot_path = ""
        clean_wavelet_path = ""
        clean_wavelet_grayscale_path = ""
        
        if comprehensive_plot:
            # 生成综合图像（时域、频域和时频域）
            comprehensive_plot_path = self.plot_comprehensive(
                signal, 
                signal_id=signal_id,
                signal_type=signal_type,
                f0=f0_wavelet, 
                f1=f1_wavelet,
            )
            result['comprehensive_plot_path'] = comprehensive_plot_path
    
        # 小波分析
        if analyze_wavelet:
            coefficients, frequencies = self.wavelet_transform(np.real(signal), f0_wavelet, f1_wavelet)
            
            # 生成小波图像
            if clean_wavelet:
                size = (6, 6)
            else:
                size = (10, 6)
                
            wavelet_plot_path, clean_wavelet_grayscale_path = self.plot_wavelet(
                coefficients, 
                frequencies, 
                signal_id=signal_id,
                signal_type=signal_type,
                clean=clean_wavelet,
                figsize=size,
                target_size=wavelet_target_size,
                output_format=wavelet_output_format
            )
            
            if clean_wavelet:
                clean_wavelet_path = wavelet_plot_path
            else:
                clean_wavelet_path = ""
                
            result['wavelet_plot_path'] = wavelet_plot_path
            result['clean_wavelet_plot_path'] = clean_wavelet_path
            result['clean_wavelet_grayscale_path'] = clean_wavelet_grayscale_path
            result['wavelet_coeff'] = coefficients
            result['wavelet_freq'] = frequencies
        
        # 存储到数据库
        db_record = {
            'signal_type': signal_type,
            'params': params,
            'snr_db': snr_db,
            'sir_db': sir_db,
            'file_path_base': base_path,
            'comprehensive_plot_path': comprehensive_plot_path,
            'wavelet_plot_path': wavelet_plot_path,
            'clean_wavelet_plot_path': clean_wavelet_path,
            'clean_wavelet_grayscale_path': clean_wavelet_grayscale_path
        }
        
        # 存储记录并获取数据库ID
        db_id = self._store_record(db_record)
        result['db_id'] = db_id
        
        return result
    
    def batch_generate(self, config_list: List[Dict[str, Any]], output_dir: Optional[str] = None, 
                      progress_callback: Optional[callable] = None,
                      default_config: Optional[Dict[str,Any]] = None) -> List[Dict[str, Any]]:
        """
        批量生成信号并进行分析
        
        参数:
            config_list: 配置列表，每个元素为字典:
                {
                    'type': 信号类型,
                    'params': 参数,
                    'save_format': 保存格式 (可选),
                    'generate_plots': 是否生成图像 (可选),
                    'analyze_wavelet': 是否进行小波分析 (可选),
                    'snr_db': 信噪比 (dB, 可选),
                    'sir_db': 信号与干扰比 (dB, 可选),
                    'save_data': 是否保存数据 (可选),
                    'clean_wavelet': 是否生成纯净小波图 (可选),
                    'comprehensive_plot': 是否生成综合图像 (可选),
                    'add_radar_signal': 是否将干扰叠加到雷达信号上 (可选),
                    'wavelet_target_size': 小波图像目标尺寸 (可选, 如 (256, 256)),
                    'wavelet_output_format': 小波图像输出格式 (可选, 'color', 'grayscale', 'both')
                }
            output_dir: 输出目录
            progress_callback: 进度回调函数，用于报告生成进度
            default_config:批处理默认参数表
        返回:
            结果列表
        """
        # 使用默认配置或自定义配置
        config_default = default_config or self.DEFAULT_CONFIG

        # 进入静默模式
        self.set_silent_mode(True)

        results = []
        total = len(config_list)
        
        for i, config in enumerate(config_list):
            if progress_callback:
                progress_callback(i, total, f"处理信号 {i+1}/{total}: {config.get('type', 'unknown')}")
            else:
                print(f"处理信号 {i+1}/{total}: {config.get('type', 'unknown')}")

            # 合并配置：默认配置 + 当前配置，当前配置会覆盖默认配置
            merged_config = {**config_default, **config}

            try:
                result = self.generate_and_analyze(
                    signal_type=merged_config['type'],
                    params=merged_config.get('params', {}),
                    output_dir=output_dir,
                    save_format=merged_config['save_format'],
                    analyze_wavelet=merged_config['analyze_wavelet'],
                    snr_db=merged_config['snr_db'],
                    sir_db=merged_config['sir_db'],
                    save_data=merged_config['save_data'],
                    clean_wavelet=merged_config['clean_wavelet'],
                    comprehensive_plot=merged_config['comprehensive_plot'],
                    show_plots=False,
                    add_radar_signal=merged_config['add_radar_signal'],
                    wavelet_target_size=merged_config['wavelet_target_size'],
                    wavelet_output_format=merged_config['wavelet_output_format']
                )

            except Exception as e:
                print(f"生成信号 { merged_config['type']} 失败: {str(e)}")
                # 可以选择继续处理其他信号或抛出异常
                continue
            
            results.append(result)
        
        return results


# ====================
# 使用示例
# ====================

def main():
    """主函数，演示如何使用信号生成器"""
    # 初始化生成器，设置雷达信号参数
    generator = RadarJammingGenerator(
        sample_rate=20e6, 
        duration=200e-6,  # 增加持续时间以容纳多个脉冲
        radar_params= {
            'f0': 1e6,           # 起始频率 1MHz
            'Kam': (5e6 - 1e6) / 30e-6,  # 调制斜率 (Hz/s)
            'pulse_width': 30e-6, # 脉冲宽度 30μs
            'PRI': 50e-6,        # 脉冲重复间隔 50μs
            'amplitude': 1.0      # 信号幅度
        }
    )
    
    # 设置自定义输出目录
    custom_output_dir = "custom_signal_output_e3"
    
    # 定义批量生成配置
    batch_config = [
        {
            'type': 'radar',
            'params': {
                'f0': 1e6,
                'Kam': (5e6 - 1e6) / 30e-6,
                'pulse_width': 30e-6,
                'PRI': 50e-6,
                'amplitude': 1.0
            },
            'save_format': 'npy',
            'analyze_wavelet': True,
            'snr_db': 20,  # 添加20dB的信噪比
            'save_data': False,  # 不保存数据
            'clean_wavelet': True,  # 生成纯净小波图
            'comprehensive_plot': False,  # 生成综合图像
            'wavelet_target_size': (256, 256),  # 设置小波图像尺寸为256x256
            'wavelet_output_format': 'both'  # 生成彩色和灰度两种格式
        },
        {
            'type': 'range_deception',
            'params': {
                'delay_time': 10e-6,  # 延迟时间 10μs
                'jamming_power': 1.2, # 干扰功率（雷达信号的1.2倍）
                'add_radar_signal': True # 是否将干扰叠加到雷达信号上
            },
            'save_format': 'npy',
            'analyze_wavelet': True,
            'snr_db': 20,  # 添加20dB的信噪比
            'save_data': False,  # 不保存数据
            'clean_wavelet': True,  # 生成纯净小波图
            'comprehensive_plot': True,  # 生成综合图像
            'wavelet_target_size': (256, 256),  # 设置小波图像尺寸为256x256
            'wavelet_output_format': 'both'  # 生成彩色和灰度两种格式
        },
        {
            'type': 'velocity_deception',
            'params': {
                'doppler_shift': 100e3,  # 多普勒频移 (100kHz)
                'jamming_power': 1.5,    # 干扰功率（雷达信号的1.5倍）
                'add_radar_signal': True # 是否将干扰叠加到雷达信号上
            },
            'analyze_wavelet': True,
            'snr_db': 20,  # 添加20dB的信噪比
            'clean_wavelet': True,  # 生成纯净小波图
            'wavelet_target_size': (256, 256),  # 设置小波图像尺寸为256x256
            'wavelet_output_format': 'both'  # 生成彩色和灰度两种格式
        },
    ]
    
    # 进度回调函数
    def progress_callback(current, total, message):
        print(f"{message} (进度: {current}/{total})")
    
    # 批量生成信号
    results = generator.batch_generate(
        batch_config,
        output_dir=custom_output_dir,
        progress_callback=progress_callback
    )
    
    # 打印结果摘要
    print("\n信号生成完成！结果摘要:")
    print("=" * 100)
    # print(f"{'序号':<5}{'信号ID':<38}{'信号类型':<20}{'彩色小波图路径':<50}{'灰度小波图路径':<50}")
    print(f"{'序号':<5}{'信号ID':<38}{'信号类型':<20}{'灰度小波图路径':<50}")
    print("-" * 100)
    
    for i, result in enumerate(results):
        # clean_wavelet = result.get('clean_wavelet_plot_path', 'N/A')
        clean_wavelet_grayscale = result.get('clean_wavelet_grayscale_path', 'N/A')
        # print(f"{i+1:<5}{result['signal_id']:<38}{result['signal_type']:<20}{clean_wavelet:<50}{clean_wavelet_grayscale:<50}")
        print(f"{i+1:<5}{result['signal_id']:<38}{result['signal_type']:<20}{clean_wavelet_grayscale:<50}")
    
    print("=" * 100)

def create_batch_config():
    """创建批量生成配置的辅助函数"""
    # 基础雷达信号配置
    radar_base = {
        'f0': 1e6,
        'Kam': (5e6 - 1e6) / 30e-6,
        'pulse_width': 40e-6,
        'PRI': 50e-6,
        'amplitude': 1.0
    }
    
    # 干扰类型生成器
    def jam_config(jam_type, params):
        return {'type': jam_type, 'params': params}
    

    # 创建配置列表
    configs = []
    
    # 1. 生成基础雷达信号（不同SNR）
    for snr in [10, 15, 20]:
        configs.append({
            'type': 'radar',
            'params': radar_base,
            'snr_db': snr
        })
    
    # 2. 距离欺骗干扰（不同延迟时间）
    for delay in [5e-6, 10e-6, 15e-6, 20e-6]:
        configs.append(jam_config('range_deception', {
            'delay_time': delay,
            'jamming_power': 1.2
        }))
    
    # 3. 速度欺骗干扰（不同多普勒频移）
    for doppler in [50e3, 100e3, 150e3, 200e3]:
        configs.append(jam_config('velocity_deception', {
            'doppler_shift': doppler,
            'jamming_power': 1.5
        }))
    
    # 4. 复合干扰（多种组合）
    composite_params = [
        {'center_freq': 3e6, 'freq_deviation': 1e6, 'jamming_power': 0.8},
        {'center_freq': 4e6, 'freq_deviation': 1.5e6, 'jamming_power': 1.0}
    ]
    for power in [1.0, 1.2]:
        configs.append({
            'type': 'composite',
            'params': {
                'components': [
                    {'type': 'noise_fm', 'params': composite_params[0]},
                    {'type': 'velocity_deception', 'params': {'doppler_shift': 150e3, 'jamming_power': power}}
                ]
            }
        })
    
    return configs

def test():
    generator = RadarJammingGenerator(sample_rate=20e6, duration=200e-6)

    custom_output_dir = "custom_signal_output_ex"

    batch_config = [
        {
            'type': 'radar',
            'params': {
                'f0': 1e6,
                'Kam': (5e6 - 1e6) / 30e-6,
                'pulse_width': 30e-6,
                'PRI': 50e-6,
                'amplitude': 1.0
            },
            'save_format': 'npy',
            'analyze_wavelet': True,
            'snr_db': 20,  # 添加20dB的信噪比
            'save_data': False,  # 不保存数据
            'clean_wavelet': True,  # 生成纯净小波图
            'comprehensive_plot': True,  # 生成综合图像
            'wavelet_target_size': (256, 256),  # 设置小波图像尺寸为256x256
            'wavelet_output_format': 'both'  # 生成彩色和灰度两种格式
        }
    ]

    # 进度回调函数
    def progress_callback(current, total, message):
        print(f"{message} (进度: {current}/{total})")
    
    # 批量生成信号
    results = generator.batch_generate(
        batch_config,
        output_dir=custom_output_dir,
        progress_callback=progress_callback
    )
    
    # 打印结果摘要
    print("\n信号生成完成！结果摘要:")
    print("=" * 100)
    # print(f"{'序号':<5}{'信号ID':<38}{'信号类型':<20}{'彩色小波图路径':<50}{'灰度小波图路径':<50}")
    print(f"{'序号':<5}{'信号ID':<38}{'信号类型':<20}{'灰度小波图路径':<50}")
    print("-" * 100)
    
    for i, result in enumerate(results):
        # clean_wavelet = result.get('clean_wavelet_plot_path', 'N/A')
        clean_wavelet_grayscale = result.get('clean_wavelet_grayscale_path', 'N/A')
        # print(f"{i+1:<5}{result['signal_id']:<38}{result['signal_type']:<20}{clean_wavelet:<50}{clean_wavelet_grayscale:<50}")
        print(f"{i+1:<5}{result['signal_id']:<38}{result['signal_type']:<20}{clean_wavelet_grayscale:<50}")
    
    print("=" * 100)
    
if __name__ == "__main__":
    main()

    # generator = RadarJammingGenerator(sample_rate=100e6, duration=40e-6)

    # config_list = create_batch_config()
    
    # # 使用自定义默认配置
    # custom_default = {
    #     'save_format': 'npy',
    #     'generate_plots': False,
    #     'analyze_wavelet': True,
    #     'snr_db': 10,
    #     'sir_db': None,
    #     'save_data': False,
    #     'clean_wavelet': True,
    #     'comprehensive_plot': False,
    #     'add_radar_signal': True,
    #     'wavelet_target_size': (256, 256),
    #     'wavelet_output_format': 'grayscale'
    # }
    
    # results = generator.batch_generate(
    #     config_list,
    #     output_dir="CNN_radar_output",
    #     default_config=custom_default
    # )