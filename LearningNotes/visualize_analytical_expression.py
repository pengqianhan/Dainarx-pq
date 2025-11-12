"""
可视化 analyticalExpression 的处理流程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
import io

# 设置 UTF-8 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色方案
color_input = '#E8F4F8'
color_step1 = '#FFE6E6'
color_step2 = '#FFF4E6'
color_step3 = '#E6F7FF'
color_output = '#E6FFE6'

# 定义框的样式
def draw_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black',
                         facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize,
            weight='bold',
            wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->,head_width=0.3,head_length=0.3',
                           linewidth=2,
                           color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 标题
ax.text(5, 9.5, 'analyticalExpression 处理流程', 
        ha='center', fontsize=16, weight='bold')

# 输入
draw_box(ax, 3.5, 8.5, 3, 0.6, 
         "输入: 'x[?] * x[?]; x0:x[?] * x_[?]'\nvar_num=2, order=2",
         color_input, fontsize=9)

# 步骤 0: 分割
draw_arrow(ax, 5, 8.5, 5, 8.0)
draw_box(ax, 3.5, 7.5, 3, 0.5,
         "步骤0: 按分号分割",
         '#F0F0F0', fontsize=9)

# 分割后的结果
draw_arrow(ax, 5, 7.5, 5, 7.0)
draw_box(ax, 2.5, 6.3, 2, 0.7,
         "表达式1:\n'x[?] * x[?]'",
         color_input, fontsize=8)
draw_box(ax, 5.5, 6.3, 2, 0.7,
         "表达式2:\n'x0:x[?] * x_[?]'",
         color_input, fontsize=8)

# 处理 x0
ax.text(1.5, 5.8, '处理变量 x0', fontsize=11, weight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# 步骤 1: 提取有效表达式
draw_arrow(ax, 3.5, 6.3, 2, 5.8)
draw_box(ax, 0.5, 5.2, 3, 0.6,
         "步骤1: extractValidExpression",
         color_step1, fontsize=9)
ax.text(0.7, 4.9, "结果: ['x[0]*x[0]', 'x[0]*x_[0]']",
        fontsize=7, style='italic')

# 步骤 2: 展开阶数
draw_arrow(ax, 2, 5.2, 2, 4.5)
draw_box(ax, 0.5, 3.9, 3, 0.6,
         "步骤2: unfoldDigit",
         color_step2, fontsize=9)
ax.text(0.7, 3.5, "结果: ['x[0]*x[0]', 'x[1]*x[1]',\n       'x[0]*x_[0]', 'x[1]*x_[1]']",
        fontsize=6.5, style='italic')

# 步骤 3: 展开变量
draw_arrow(ax, 2, 3.9, 2, 3.0)
draw_box(ax, 0.5, 2.4, 3, 0.6,
         "步骤3: unfoldItem",
         color_step3, fontsize=9)
ax.text(0.7, 1.9, "结果: ['x[0][0]*x[0][0]', 'x[0][1]*x[0][1]',\n"+
               "       'x[0][0]*x[1][0]', 'x[0][1]*x[1][1]']",
        fontsize=6, style='italic')

# 步骤 4: 生成 lambda
draw_arrow(ax, 2, 2.4, 2, 1.5)
draw_box(ax, 0.5, 0.9, 3, 0.6,
         "步骤4: 转换为 lambda 函数",
         color_output, fontsize=9)
ax.text(0.7, 0.5, "x0 的 4 个特征函数",
        fontsize=7, weight='bold', style='italic')

# 处理 x1
ax.text(7.5, 5.8, '处理变量 x1', fontsize=11, weight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# 步骤 1
draw_arrow(ax, 6.5, 6.3, 8, 5.8)
draw_box(ax, 6.5, 5.2, 3, 0.6,
         "步骤1: extractValidExpression",
         color_step1, fontsize=9)
ax.text(6.7, 4.9, "结果: ['x[0]*x[0]']  (无 x0: 条件)",
        fontsize=7, style='italic')

# 步骤 2
draw_arrow(ax, 8, 5.2, 8, 4.5)
draw_box(ax, 6.5, 3.9, 3, 0.6,
         "步骤2: unfoldDigit",
         color_step2, fontsize=9)
ax.text(6.7, 3.5, "结果: ['x[0]*x[0]', 'x[1]*x[1]']",
        fontsize=6.5, style='italic')

# 步骤 3
draw_arrow(ax, 8, 3.9, 8, 3.0)
draw_box(ax, 6.5, 2.4, 3, 0.6,
         "步骤3: unfoldItem",
         color_step3, fontsize=9)
ax.text(6.7, 1.9, "结果: ['x[1][0]*x[1][0]', 'x[1][1]*x[1][1]']",
        fontsize=6, style='italic')

# 步骤 4
draw_arrow(ax, 8, 2.4, 8, 1.5)
draw_box(ax, 6.5, 0.9, 3, 0.6,
         "步骤4: 转换为 lambda 函数",
         color_output, fontsize=9)
ax.text(6.7, 0.5, "x1 的 2 个特征函数",
        fontsize=7, weight='bold', style='italic')

# 最终输出
draw_arrow(ax, 2, 0.9, 5, 0.3)
draw_arrow(ax, 8, 0.9, 5, 0.3)
draw_box(ax, 3, 0, 4, 0.3,
         "返回: (res, res_order)",
         '#FFD700', fontsize=10)

# 添加图例
legend_elements = [
    mpatches.Patch(color=color_input, label='输入/中间结果'),
    mpatches.Patch(color=color_step1, label='步骤1: 提取'),
    mpatches.Patch(color=color_step2, label='步骤2: 展开阶数'),
    mpatches.Patch(color=color_step3, label='步骤3: 展开变量'),
    mpatches.Patch(color=color_output, label='步骤4: 输出'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('result/analyticalExpression_flow.png', dpi=300, bbox_inches='tight')
print("✓ 流程图已保存到: result/analyticalExpression_flow.png")

# 创建第二个图：数据结构示例
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# 左图：输入数据结构
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('输入数据结构 x[i][j]', fontsize=14, weight='bold', pad=20)

# 绘制二维数组
cell_width = 1.5
cell_height = 0.8
start_x = 1.5
start_y = 6

# 表头
ax1.text(start_x - 0.8, start_y + cell_height/2, 'x:', 
         fontsize=11, weight='bold', va='center')

# 时间标签
time_labels = ['t-3', 't-2', 't-1', 't']
for j, label in enumerate(time_labels):
    ax1.text(start_x + j*cell_width + cell_width/2, start_y + cell_height + 0.3,
             label, ha='center', fontsize=9, weight='bold')

# 变量标签和数据
var_data = [
    ['1.0', '2.0', '3.0', '4.0'],  # x0
    ['5.0', '6.0', '7.0', '8.0']   # x1
]

for i, row_data in enumerate(var_data):
    # 变量标签
    ax1.text(start_x - 0.8, start_y - i*cell_height - cell_height/2,
             f'x{i}', fontsize=11, weight='bold', va='center')
    
    # 数据单元格
    for j, value in enumerate(row_data):
        rect = FancyBboxPatch((start_x + j*cell_width, start_y - i*cell_height - cell_height),
                              cell_width*0.9, cell_height*0.9,
                              boxstyle="round,pad=0.05",
                              edgecolor='black',
                              facecolor='lightblue',
                              linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(start_x + j*cell_width + cell_width/2, 
                start_y - i*cell_height - cell_height/2,
                value, ha='center', va='center', fontsize=10)

# 索引示例
ax1.text(5, 3.5, '索引示例:', fontsize=11, weight='bold')
examples = [
    'x[0][0] = 1.0  (x0 在 t-3)',
    'x[0][-1] = 4.0  (x0 在 t，最新)',
    'x[1][1] = 6.0  (x1 在 t-2)',
    'x[1][-2] = 7.0  (x1 在 t-1)',
]
for k, ex in enumerate(examples):
    ax1.text(5, 3.0 - k*0.5, ex, fontsize=9, family='monospace')

# 右图：函数调用示例
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('生成的 Lambda 函数示例', fontsize=14, weight='bold', pad=20)

y_pos = 8.5
# 函数1
draw_box(ax2, 1, y_pos, 8, 0.8,
         "函数: lambda x: x[0][0] * x[0][0]",
         '#FFE6E6', fontsize=10)
ax2.text(1.2, y_pos - 0.5, "计算: x[0][0] * x[0][0] = 1.0 * 1.0 = 1.0",
         fontsize=9, style='italic')
ax2.text(1.2, y_pos - 0.9, "含义: x0 在 t-3 时刻的平方",
         fontsize=9, color='blue')

y_pos -= 2
# 函数2
draw_box(ax2, 1, y_pos, 8, 0.8,
         "函数: lambda x: x[0][1] * x[0][1]",
         '#FFF4E6', fontsize=10)
ax2.text(1.2, y_pos - 0.5, "计算: x[0][1] * x[0][1] = 2.0 * 2.0 = 4.0",
         fontsize=9, style='italic')
ax2.text(1.2, y_pos - 0.9, "含义: x0 在 t-2 时刻的平方",
         fontsize=9, color='blue')

y_pos -= 2
# 函数3
draw_box(ax2, 1, y_pos, 8, 0.8,
         "函数: lambda x: x[0][0] * x[1][0]",
         '#E6F7FF', fontsize=10)
ax2.text(1.2, y_pos - 0.5, "计算: x[0][0] * x[1][0] = 1.0 * 5.0 = 5.0",
         fontsize=9, style='italic')
ax2.text(1.2, y_pos - 0.9, "含义: x0 和 x1 在 t-3 时刻的交叉项",
         fontsize=9, color='blue')

y_pos -= 2
# 函数4
draw_box(ax2, 1, y_pos, 8, 0.8,
         "函数: lambda x: x[0][1] * x[1][1]",
         '#E6FFE6', fontsize=10)
ax2.text(1.2, y_pos - 0.5, "计算: x[0][1] * x[1][1] = 2.0 * 6.0 = 12.0",
         fontsize=9, style='italic')
ax2.text(1.2, y_pos - 0.9, "含义: x0 和 x1 在 t-2 时刻的交叉项",
         fontsize=9, color='blue')

plt.tight_layout()
plt.savefig('result/analyticalExpression_data_structure.png', dpi=300, bbox_inches='tight')
print("✓ 数据结构图已保存到: result/analyticalExpression_data_structure.png")

print("\n所有可视化完成！")

