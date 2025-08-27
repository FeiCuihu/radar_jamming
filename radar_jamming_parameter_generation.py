from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from signal_radar_jamming import RadarJammingGenerator


@dataclass
class SignalConfig:
    """信号配置类"""
    signal_type: str
    params: Dict[str, Any]
    snr_db: Optional[float] = 10
    sir_db: Optional[float] = None
    save_format: str = "npy"
    analyze_wavelet: bool = True
    save_data: bool = False
    clean_wavelet: bool = True
    comprehensive_plot: bool = True
    add_radar_signal: bool = True
    wavelet_target_size: Optional[Tuple[int, int]] = (256, 256)
    wavelet_output_format: str = "grayscale"
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            'type': self.signal_type,
            'params': self.params,
            'snr_db': self.snr_db,
            'sir_db': self.sir_db,
            'save_format': self.save_format,
            'analyze_wavelet': self.analyze_wavelet,
            'save_data': self.save_data,
            'clean_wavelet': self.clean_wavelet,
            'comprehensive_plot': self.comprehensive_plot,
            'add_radar_signal': self.add_radar_signal,
            'wavelet_target_size': self.wavelet_target_size,
            'wavelet_output_format': self.wavelet_output_format
        }

@dataclass
class BatchConfig:
    """批量配置参数类"""
    # 雷达基础参数
    radar_f0: float = 1e6
    radar_Kam:float = (5e6 - 1e6) / 30e-6
    radar_f1: float = 5e6
    radar_pulse_width: float = 40e-6
    radar_pri: float = 50e-6
    radar_amplitude: float = 1.0
    
    # SNR参数
    snr_values: List[float] = field(default_factory=lambda: [10, 15, 20])
    
    # 距离欺骗参数
    range_delays: List[float] = field(default_factory=lambda: [5e-6, 10e-6, 15e-6, 20e-6])
    range_powers: List[float] = field(default_factory=lambda: [1.2])
    
    # 速度欺骗参数
    velocity_dopplers: List[float] = field(default_factory=lambda: [50e3, 100e3, 150e3, 200e3])
    velocity_powers: List[float] = field(default_factory=lambda: [1.5])
    
    # 噪声调频参数
    noise_fm_center_freqs: List[float] = field(default_factory=lambda: [3e6, 4e6])
    noise_fm_deviations: List[float] = field(default_factory=lambda: [1e6, 1.5e6])
    noise_fm_powers: List[float] = field(default_factory=lambda: [0.8, 1.0])
    
    # 噪声调幅参数
    noise_am_center_freqs: List[float] = field(default_factory=lambda: [3e6, 4e6])
    noise_am_bandwidths: List[float] = field(default_factory=lambda: [1e6, 1.5e6])
    noise_am_powers: List[float] = field(default_factory=lambda: [0.8, 1.0])
    
    # 密集假目标参数
    dense_false_counts: List[int] = field(default_factory=lambda: [5, 10])
    dense_false_powers: List[float] = field(default_factory=lambda: [0.8, 1.0])
    
    # 复合干扰参数
    composite_powers: List[float] = field(default_factory=lambda: [1.0, 1.2])


def create_batch_config(config: BatchConfig = None)-> List[Dict[str, Any]]:
    """
    创建批量生成配置
    
    参数:
        config: 批量配置参数，如果为None则使用默认配置
        
    返回:
        配置字典列表
    """
    if config is None:
        config = BatchConfig()

    # 基础雷达信号配置 - 修复Kam计算
    radar_base = {
        'f0': 1e6,
        'Kam': (5e6 - 1e6) / 40e-6,  # 使用正确的脉冲宽度
        'pulse_width': 40e-6,
        'PRI': 50e-6,
        'amplitude': 1.0
    }
    
    # 创建配置列表
    configs = []
    
    # 1. 生成基础雷达信号（不同SNR）
    for snr in config.snr_values:
        signal_config = SignalConfig(
            signal_type='radar',
            params=radar_base,
            snr_db=snr
        )
        configs.append(signal_config.to_dict())
    
    # 2. 距离欺骗干扰
    for delay in config.range_delays:
        for power in config.range_powers:
            signal_config = SignalConfig(
                signal_type='range_deception',
                params={'delay_time': delay, 'jamming_power': power}
            )
            configs.append(signal_config.to_dict())
    
    # 3. 速度欺骗干扰
    for doppler in config.velocity_dopplers:
        for power in config.velocity_powers:
            signal_config = SignalConfig(
                signal_type='velocity_deception',
                params={'doppler_shift': doppler, 'jamming_power': power}
            )
            configs.append(signal_config.to_dict())
    
    # 4. 噪声调频干扰
    for center_freq in config.noise_fm_center_freqs:
        for deviation in config.noise_fm_deviations:
            for power in config.noise_fm_powers:
                signal_config = SignalConfig(
                    signal_type='noise_fm',
                    params={
                        'center_freq': center_freq,
                        'freq_deviation': deviation,
                        'jamming_power': power
                    }
                )
                configs.append(signal_config.to_dict())
    
    # 5. 噪声调幅干扰
    for center_freq in config.noise_am_center_freqs:
        for bandwidth in config.noise_am_bandwidths:
            for power in config.noise_am_powers:
                signal_config = SignalConfig(
                    signal_type='noise_am',
                    params={
                        'center_freq': center_freq,
                        'bandwidth': bandwidth,
                        'jamming_power': power
                    }
                )
                configs.append(signal_config.to_dict())
    
    # 6. 密集假目标干扰
    for count in config.dense_false_counts:
        for power in config.dense_false_powers:
            signal_config = SignalConfig(
                signal_type='dense_false',
                params={'num_targets': count, 'jamming_power': power}
            )
            configs.append(signal_config.to_dict())
    
    # 7. 复合干扰
    for power in config.composite_powers:
        signal_config = SignalConfig(
            signal_type='composite',
            params={
                'components': [
                    {
                        'type': 'noise_fm',
                        'params': {
                            'center_freq': 3e6,
                            'freq_deviation': 1e6,
                            'jamming_power': 0.8
                        }
                    },
                    {
                        'type': 'velocity_deception',
                        'params': {
                            'doppler_shift': 150e3,
                            'jamming_power': power
                        }
                    }
                ]
            }
        )
        configs.append(signal_config.to_dict())
    
    return configs

if __name__ == "__main__":
    # 创建自定义批量配置
    

    # custom_batch_config = BatchConfig(
    #     snr_values=[10, 20, 30],  # 只生成3种SNR的雷达信号
    #     range_delays=[5e-6, 15e-6],  # 只生成2种延迟的距离欺骗
    #     velocity_dopplers=[100e3, 200e3],  # 只生成2种多普勒的速度欺骗
    #     # 其他参数使用默认值
    # )

    custom_batch_config = BatchConfig()

    config_list = create_batch_config(custom_batch_config)
    
    # for i, config in enumerate(config_list):
    #     print(f"{i+1:<5}{config}")

    # 使用自定义默认配置
    custom_default = {
        'save_format': 'npy',
        'generate_plots': False,
        'analyze_wavelet': False,
        'snr_db': 10,
        'sir_db': 10,
        'save_data': False,
        'clean_wavelet': False,
        'comprehensive_plot': False,
        'add_radar_signal': True,
        'wavelet_target_size': (256, 256),
        'wavelet_output_format': 'grayscale'
    }

    generator = RadarJammingGenerator(sample_rate=200e6, duration=20e-6)

    results = generator.batch_generate(
        config_list,
        output_dir="radar_output",
        default_config=custom_default
    )
    
    print(f"生成了 {len(results)} 个信号配置")