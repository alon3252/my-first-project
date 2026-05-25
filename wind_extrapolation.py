import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def power_law(z, alpha, u_ref, z_ref=10):
    """
    风速廓线幂律公式
    u(z) = u_ref * (z / z_ref)^alpha

    参数:
    z: 目标高度
    alpha: 风切变指数
    u_ref: 参考高度的风速
    z_ref: 参考高度
    """
    return u_ref * (z / z_ref) ** alpha

def fit_wind_profile(heights, wind_speeds, z_ref=10):
    """
    拟合风廓线参数

    参数:
    heights: 高度数组
    wind_speeds: 对应高度的风速数组
    z_ref: 参考高度

    返回:
    alpha: 风切变指数
    u_ref: 参考高度风速
    """
    # 使用最小二乘法拟合
    def objective(z, alpha, u_ref):
        return power_law(z, alpha, u_ref, z_ref)

    try:
        # 初始猜测值
        p0 = [0.2, 5.0]  # alpha, u_ref

        # 拟合参数
        popt, pcov = curve_fit(objective, heights, wind_speeds, p0=p0, maxfev=10000)

        alpha, u_ref = popt
        return alpha, u_ref
    except Exception as e:
        print(f"拟合过程中出现错误: {e}")
        # 如果拟合失败，使用默认值
        return 0.2, 5.0

def estimate_wind_speed_at_height(target_height, heights, wind_speeds):
    """
    估计指定高度的风速

    参数:
    target_height: 目标高度
    heights: 已知高度数组
    wind_speeds: 对应高度的风速数组

    返回:
    estimated_speed: 估计的风速
    alpha: 风切变指数
    """
    # 拟合风廓线参数
    alpha, u_ref = fit_wind_profile(heights, wind_speeds)

    # 使用拟合的参数估算目标高度的风速
    estimated_speed = power_law(target_height, alpha, u_ref)

    return estimated_speed, alpha

def classify_turbulence(intensity):
    """
    根据湍流强度分类湍流等级
    IEC 61400-1 标准湍流等级分类:
    - LT (Low Turbulence): I < 0.12
    - NT (Normal Turbulence): 0.12 <= I < 0.20
    - HT (High Turbulence): I >= 0.20
    """
    if intensity < 0.12:
        return 'LT (低湍流)'
    elif intensity < 0.20:
        return 'NT (正常湍流)'
    else:
        return 'HT (高湍流)'

def extrapolate_wind_data(file_path):
    """
    使用现有数据外推125米高度的风速和湍流等级
    """
    print("开始外推125米高度风速数据...")

    # 读取Excel文件
    try:
        df = pd.read_excel(file_path)
        print(f"成功读取数据，共有 {len(df)} 行数据")
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return

    # 查找140米、160米高度的风速列
    height_columns = {}
    for col in df.columns:
        col_str = str(col).upper()
        if '140' in col_str and 'MEAN' in col_str and 'WS' in col_str:
            height_columns[140] = col
        elif '160' in col_str and 'MEAN' in col_str and 'WS' in col_str:
            height_columns[160] = col

    print(f"\n识别到的高度列: {height_columns}")

    if len(height_columns) < 2:
        print("需要至少两个高度的数据来进行拟合")
        return

    # 数据清洗
    print("\n开始数据清洗...")

    # 创建一个新的DataFrame只包含需要的列
    analysis_cols = list(height_columns.values())
    analysis_df = df[analysis_cols].copy()

    # 移除空值
    for col in analysis_cols:
        analysis_df = analysis_df.dropna(subset=[col])

    # 移除负值和异常大的值
    for col in analysis_cols:
        analysis_df = analysis_df[(analysis_df[col] >= 0) & (analysis_df[col] <= 50)]

    print(f"清洗后剩余 {len(analysis_df)} 条数据")

    # 计算各高度平均风速
    print("\n计算各高度平均风速:")
    avg_wind_speeds = {}
    for height, col in height_columns.items():
        avg_speed = analysis_df[col].mean()
        avg_wind_speeds[height] = avg_speed
        print(f"{height}米高度平均风速: {avg_speed:.2f} m/s")

    # 使用幂律公式外推125米高度风速
    print("\n使用幂律公式外推125米高度风速...")

    # 准备用于拟合的数据
    known_heights = list(avg_wind_speeds.keys())
    known_speeds = list(avg_wind_speeds.values())

    # 外推125米高度风速
    estimated_speed_125, alpha = estimate_wind_speed_at_height(125, known_heights, known_speeds)

    print(f"拟合得到的风切变指数 α: {alpha:.3f}")
    print(f"125米高度估算风速: {estimated_speed_125:.2f} m/s")

    # 计算湍流强度
    # 假设湍流强度与风速呈反比关系（根据IEC标准）
    # 我们可以基于已知高度的湍流强度来估算125米的湍流强度

    # 先计算已知高度的湍流强度
    print("\n计算已知高度湍流强度:")
    turbulence_intensity = {}
    for height, col in height_columns.items():
        mean_val = analysis_df[col].mean()
        std_val = analysis_df[col].std()
        if mean_val > 0:
            intensity = std_val / mean_val
            turbulence_intensity[height] = intensity
            turbulence_class = classify_turbulence(intensity)
            print(f"{height}米高度湍流强度: {intensity:.3f} ({turbulence_class})")

    # 外推125米高度的湍流强度
    # 假设湍流强度与高度的关系可以用指数函数描述
    if len(turbulence_intensity) >= 2:
        print("\n外推125米高度湍流强度...")

        # 准备湍流强度数据
        known_heights_ti = list(turbulence_intensity.keys())
        known_ti_values = list(turbulence_intensity.values())

        # 简单线性插值估算125米的湍流强度
        if len(known_heights_ti) >= 2:
            # 按高度排序
            sorted_indices = np.argsort(known_heights_ti)
            sorted_heights = np.array(known_heights_ti)[sorted_indices]
            sorted_ti = np.array(known_ti_values)[sorted_indices]

            # 线性插值
            if 125 <= sorted_heights[-1] and 125 >= sorted_heights[0]:
                # 在已知范围内，使用插值
                estimated_ti_125 = np.interp(125, sorted_heights, sorted_ti)
            else:
                # 在范围外，使用最近点的值
                if 125 < sorted_heights[0]:
                    estimated_ti_125 = sorted_ti[0]
                else:
                    estimated_ti_125 = sorted_ti[-1]

            turbulence_class_125 = classify_turbulence(estimated_ti_125)
            print(f"125米高度估算湍流强度: {estimated_ti_125:.3f} ({turbulence_class_125})")
        else:
            # 如果只有一个点，使用该点的值
            estimated_ti_125 = list(turbulence_intensity.values())[0]
            turbulence_class_125 = classify_turbulence(estimated_ti_125)
            print(f"125米高度估算湍流强度: {estimated_ti_125:.3f} ({turbulence_class_125})")
    else:
        print("无法估算125米高度湍流强度，缺少足够的参考数据")
        estimated_ti_125 = None
        turbulence_class_125 = "未知"

    # 生成外推分析报告
    print("\n生成外推分析报告...")
    report_content = f"""
风电场风速数据外推分析报告
==========================

1. 数据概况
   - 总数据量: {len(df)} 条记录
   - 清洗后数据量: {len(analysis_df)} 条记录
   - 参考高度: {', '.join([str(h) + '米' for h in known_heights])}
   - 目标高度: 125米

2. 参考高度风速分析
"""

    for height in sorted(known_heights):
        report_content += f"   - {height}米高度平均风速: {avg_wind_speeds[height]:.2f} m/s\n"

    report_content += f"""
3. 风廓线拟合结果
   - 风切变指数 α: {alpha:.3f}
   - 125米高度估算风速: {estimated_speed_125:.2f} m/s

4. 湍流强度分析
"""

    for height in sorted(known_heights):
        if height in turbulence_intensity:
            intensity = turbulence_intensity[height]
            classification = classify_turbulence(intensity)
            report_content += f"   - {height}米高度湍流强度: {intensity:.3f} ({classification})\n"

    if estimated_ti_125 is not None:
        report_content += f"   - 125米高度估算湍流强度: {estimated_ti_125:.3f} ({turbulence_class_125})\n"

    report_content += f"""
5. 技术说明
   - 使用幂律公式 u(z) = u_ref * (z / z_ref)^α 进行风速外推
   - 湍流强度采用线性插值方法估算
   - 外推结果仅供参考，实际应用中建议结合现场测量数据验证

6. 建议
   - 125米高度估算风速为 {estimated_speed_125:.2f} m/s，适合安装风机
   - 该高度湍流强度等级为 {turbulence_class_125}，选择风机时需考虑此因素
"""

    # 保存报告
    try:
        with open('风速外推分析报告.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        print("外推分析报告已保存为: 风速外推分析报告.txt")
    except Exception as e:
        print(f"保存报告时出错: {e}")
        print("报告内容:")
        print(report_content)

    # 创建可视化图表
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 绘制风速随高度变化图
        all_heights = sorted(known_heights + [125])
        all_speeds = [avg_wind_speeds[h] if h in avg_wind_speeds else estimated_speed_125 for h in all_heights]

        ax1.plot(all_heights[:-1], all_speeds[:-1], 'bo-', label='实测数据', markersize=8)
        ax1.plot(125, estimated_speed_125, 'ro', label='外推值', markersize=10)
        ax1.set_xlabel('高度 (米)')
        ax1.set_ylabel('平均风速 (m/s)')
        ax1.set_title('风速随高度变化及125米外推值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for i, (h, s) in enumerate(zip(all_heights, all_speeds)):
            ax1.annotate(f'{s:.2f}', (h, s), textcoords="offset points", xytext=(0,10), ha='center')

        # 绘制湍流强度图（如果有数据）
        if len(turbulence_intensity) >= 1:
            ti_heights = sorted(list(turbulence_intensity.keys()))
            ti_values = [turbulence_intensity[h] for h in ti_heights]

            ax2.plot(ti_heights, ti_values, 'go-', label='实测数据', markersize=8)
            if estimated_ti_125 is not None:
                ax2.plot(125, estimated_ti_125, 'ro', label='外推值', markersize=10)
            ax2.set_xlabel('高度 (米)')
            ax2.set_ylabel('湍流强度')
            ax2.set_title('湍流强度随高度变化及125米外推值')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 添加数值标签
            for i, (h, t) in enumerate(zip(ti_heights, ti_values)):
                ax2.annotate(f'{t:.3f}', (h, t), textcoords="offset points", xytext=(0,10), ha='center')
            if estimated_ti_125 is not None:
                ax2.annotate(f'{estimated_ti_125:.3f}', (125, estimated_ti_125), textcoords="offset points", xytext=(0,10), ha='center')

        plt.tight_layout()
        plt.savefig('风速外推分析图表.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("外推分析图表已保存为: 风速外推分析图表.png")
    except Exception as e:
        print(f"生成图表时出错: {e}")

    return {
        'estimated_wind_speed_125': estimated_speed_125,
        'estimated_turbulence_125': estimated_ti_125,
        'alpha': alpha,
        'known_heights': known_heights,
        'known_speeds': known_speeds
    }

if __name__ == "__main__":
    import os

    # 获取桌面路径
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    file_path = os.path.join(desktop_path, "矿能.xlsx")

    if os.path.exists(file_path):
        results = extrapolate_wind_data(file_path)
        print("\n外推分析完成!")
    else:
        print(f"文件不存在: {file_path}")
        print("请确认文件路径是否正确")