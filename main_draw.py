import sys
import os

# 读取绘图
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置西文字体为新罗马字体
from matplotlib import rcParams
from matplotlib import font_manager as fm

# 自动加载所有本地字体
for font_file in os.listdir('./fonts'):
    if font_file.lower().endswith(('.ttf', '.otf')):
        fm.fontManager.addfont('./fonts/' + font_file)  # v3.5+ 推荐方法

# 字体配置
config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False  # 解决负号无法显示的问题
}
rcParams.update(config)

# 解析
def ParserCsv(csvPD):
	# 定义列表
	order_list = []
	test_times_list = []
	load_time_list = []
	cal_time_list = []

	# 遍历解析
	for i in range(len(csvPD)):
		order_list.append(csvPD["Order"][i])
		test_times_list.append(csvPD["Test-Times"][i])
		load_time_list.append(csvPD["Load-Time"][i])
		cal_time_list.append(csvPD["Cal-Time"][i])

	return order_list, test_times_list, load_time_list, cal_time_list

# 解析数据
def analyze_data(
	filepath
):
	# 定义列表
	final_order_list = []
	sum_load_time_list = []
	sum_cal_time_list = []
	sum_time_list = []
	
	# 安全解析
	try:
		# 读取CSV文件
		csvPD = pd.read_csv(filepath)
		
		# 解析
		order_list, test_times_list, load_time_list, cal_time_list = ParserCsv(csvPD=csvPD)
		
		# 定义结果列表
		res_list = []
		
		# 上一次阶数
		order_last = -1
		
		# 求和
		sum_load_time = 0.0
		sum_cal_time = 0.0
		cur_times = 0
		
		# 遍历阶数
		for i in range(0,len(order_list)):
			# 如果阶数不一样
			if order_list[i] != order_last and i > 0:
				
				# 求平均
				sum_time = (sum_load_time + sum_cal_time) / cur_times
				sum_load_time = sum_load_time / cur_times
				sum_cal_time = sum_cal_time / cur_times
				
				# 添加进入列表
				sum_load_time_list.append(sum_load_time)
				sum_cal_time_list.append(sum_cal_time)
				sum_time_list.append(sum_time)
				final_order_list.append(order_last)
				
				# 清零
				sum_load_time = 0.0
				sum_cal_time = 0.0
				cur_times = 0
				
			# 如果不是最后一次
			if i < len(order_list) - 1:
				# 如果当前次数为0
				if test_times_list[i] == 0 and test_times_list[i + 1] > 0:
					# 如果超出范围:第一次算时间不准,要剔除
					if cal_time_list[i] > cal_time_list[i + 1] * 10:
						# 传递
						order_last = order_list[i]
						continue
			
			# 如果阶数一样,则说明时间要累计
			sum_load_time = sum_load_time + load_time_list[i]
			sum_cal_time = sum_cal_time + cal_time_list[i]
			
			# 传递
			order_last = order_list[i]
			cur_times = cur_times + 1
		
		# 求平均
		sum_time = (sum_load_time + sum_cal_time) / cur_times
		sum_load_time = sum_load_time / cur_times
		sum_cal_time = sum_cal_time / cur_times
		
		# 添加进入列表
		sum_load_time_list.append(sum_load_time)
		sum_cal_time_list.append(sum_cal_time)
		sum_time_list.append(sum_time)
		final_order_list.append(order_last)
		
		# 最终的列表
		res_list = [final_order_list,sum_load_time_list,sum_cal_time_list,sum_time_list]
		
	# 读取文件异常
	except IOError:
		print("> INFO: [python] The path '{0}' cannot be found!".format(filepath))
		
		# 返回列表
		return [[0],[0],[0],[0]]
	# 退出
	else:
		print("> INFO: [python] The path '{0}' has been read successfully!".format(filepath))
		
		# 返回列表
		return res_list

# 画图
def draw_runtime(
	sub_fig_enable,
	legend_enable,
	legend_pos,
	draw_sel,
	csv_data_list,
	label_list,
	xticks_list,
	yticks_list,
	fig_id,
	dpi,
	save_path,
	title="",
	xlabel="",
	ylabel=""
):
	# 绘图参数列表
	marker_list = [
		['o',9],    # 圆圈 (Circle)
		['s',7],    # 方形 (Square)
		['^',9],    # 上三角 (Triangle up)
		['>',9],	# 星号
		['p',9],    # 五边形 (Pentagon)
		['v',9],    # 下三角 (Triangle down)
		['h',9],    # 六边形
		['D',7],    # 菱形 (Diamond)
	]
	
	# 高级配置建议（适配出版级绘图）
	params = {
		'markeredgewidth': 1.2,    # 边缘线宽
		'markeredgecolor': 'k'    # 边缘颜色
	}

	# 8种标记的推荐配色方案
	colors = [
		'#7f7f7f', 
		'#2ca02c', 
		'#ff7f0e',
		'#9467bd', 
		'#e377c2', 
		'#1f77b4', 
		'#8c564b', 
		'#d62728'
	]
	
	# 画线
	linestyles = [
		'-',	# zernipax-cpu
		'-',	# zernpy-cpu
		'-',	# zern-cpu
		'-',	# zernike-cpu
		'-.',	# zernipax-gpu
		'--',	# ours-cpu
		'--',	# ours-gpu
		'-',	# ours-fpga
	]
	
	# 线宽适配参数配置
	plt.rcParams['lines.dash_capstyle'] = 'round'  # 端点圆角处理
	plt.rcParams['lines.dash_joinstyle'] = 'bevel' # 连接处斜切处理

	# 设置画布
	plt.figure(fig_id, figsize=(7.5,8.5),dpi=dpi)
	
	# 遍历
	for i in range(0,len(csv_data_list)):
		
		# 定义横纵坐标
		x_pos = csv_data_list[i][0]
		y_pos = csv_data_list[i][draw_sel]	# 画图选择
		
		# 绘制折线图
		plt.semilogy(
			x_pos,
			y_pos,
			color=colors[i],
			linestyle=linestyles[i],
			label=label_list[i],
			marker=marker_list[i][0],
			markersize=marker_list[i][1],
			**params
		)
	
	# 设置横纵坐标
	plt.xticks(xticks_list)
	plt.yticks(yticks_list)

	# 设置标签和刻度大小
	plt.tick_params(labelsize=20)
	plt.ylabel(ylabel, fontsize=22)
	plt.xlabel(xlabel, fontsize=22)
	
	# 如果使能图例
	if legend_enable == True:
		plt.legend(ncol=2, fontsize=18, loc=legend_pos)
	
	plt.grid(ls="--", alpha=0.5)

	# 标题
	plt.title(title,fontsize=24)
	
	# 如果是子图
	if sub_fig_enable == True:
		plt.ylabel("", fontsize=18)
		ax = plt.gca()
		ax.yaxis.set_ticklabels([])	# 清除标签
		ax.tick_params(left=True)	# 保留刻度线
	
	# 保存
	plt.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
	
# 主函数
if __name__ == "__main__":

	# 根目录
	root_path = './evaluation/time'
	
	# 测试路径
	test_path_cpu = '/draw/' # '/2025-03-31_21-45-22/'
	test_path_gpu = '/draw/' # '/2025-03-31_21-29-18/'
	test_path_fpga = '/draw/'
	
	# 测试文件路径
	test_ours_cpu_filename = 'ours-cpu-time'
	test_zernipax_cpu_filename = 'zernipax-cpu-time'
	test_zernpy_cpu_filename = 'zernpy-cpu-time'
	test_zern_cpu_filename = 'zern-cpu-time'
	test_zernike_cpu_filename = 'zernike-cpu-time'
	
	test_ours_gpu_filename = 'ours-gpu-time'
	test_zernipax_gpu_filename = 'zernipax-gpu-time'
	
	test_ours_fpga_filename = 'ours-fpga-time'
	
	# 最终路径
	test_ours_cpu_path = root_path + test_path_cpu + test_ours_cpu_filename + '.txt'
	test_zernipax_cpu_path = root_path + test_path_cpu + test_zernipax_cpu_filename + '.txt'
	test_zernpy_cpu_path = root_path + test_path_cpu + test_zernpy_cpu_filename + '.txt'
	test_zern_cpu_path = root_path + test_path_cpu + test_zern_cpu_filename + '.txt'
	test_zernike_cpu_path = root_path + test_path_cpu + test_zernike_cpu_filename + '.txt'
	
	test_ours_gpu_path = root_path + test_path_gpu + test_ours_gpu_filename + '.txt'
	test_zernipax_gpu_path = root_path + test_path_gpu + test_zernipax_gpu_filename + '.txt'
	
	test_ours_fpga_path = root_path + test_path_fpga + test_ours_fpga_filename + '.txt'
	
	# 解析数据--ours-cpu
	test_ours_cpu_list = analyze_data(
		filepath=test_ours_cpu_path
	)
	
	# 解析数据--zernipax-cpu
	test_zernipax_cpu_list = analyze_data(
		filepath=test_zernipax_cpu_path
	)
	
	# 解析数据--zernpy-cpu
	test_zernpy_cpu_list = analyze_data(
		filepath=test_zernpy_cpu_path
	)
	
	# 解析数据--zern-cpu
	test_zern_cpu_list = analyze_data(
		filepath=test_zern_cpu_path
	)
	
	# 解析数据--zernike-cpu
	test_zernike_cpu_list = analyze_data(
		filepath=test_zernike_cpu_path
	)
	
	# 解析数据--ours-gpu
	test_ours_gpu_list = analyze_data(
		filepath=test_ours_gpu_path
	)
	
	# 解析数据--zernipax-gpu
	test_zernipax_gpu_list = analyze_data(
		filepath=test_zernipax_gpu_path
	)
	
	# 解析数据--ours-fpga
	test_ours_fpga_list = analyze_data(
		filepath=test_ours_fpga_path
	)
	
	# 生成列表
	csv_data_list = [
		test_zernipax_cpu_list,
		test_zernpy_cpu_list,
		test_zern_cpu_list,
		test_zernike_cpu_list,
		test_zernipax_gpu_list,
		test_ours_cpu_list,
		test_ours_gpu_list,
		test_ours_fpga_list
	]
	
	# 标签列表
	label_list = [
		'zernipax-cpu',
		'zernpy-cpu',
		'zern-cpu',
		'zernike-cpu',
		'zernipax-gpu',
		'ours-cpu',
		'ours-gpu',
		'ours-fpga'
	]
	
	# 绘图的dpi
	figure_dpi = 1000
	xticks_list = [1,4,8,12,16,20]
	yticks_list = [0.001,0.01,0.1,1,10,100,1000,10000,100000]
	
	# 绘制运行时间图
	draw_runtime(
		sub_fig_enable=True,
		legend_enable=False,
		legend_pos=(0.2,0.2),
		draw_sel=2,
		csv_data_list=csv_data_list,
		label_list=label_list,
		xticks_list=xticks_list,
		yticks_list=yticks_list,
		fig_id=0,
		dpi=figure_dpi,
		save_path='./computation_runtime.jpg',
		title="Computation times of u(ρ,θ)",
		xlabel="First n-Order",
		ylabel="Runtime (s)"
	)
	
	# 绘制加载时间图
	draw_runtime(
		sub_fig_enable=False,
		legend_enable=True,
		legend_pos=None,
		draw_sel=1,
		csv_data_list=csv_data_list,
		label_list=label_list,
		xticks_list=xticks_list,
		yticks_list=yticks_list,
		fig_id=1,
		dpi=figure_dpi,
		save_path='./load_runtime.jpg',
		title="Load times of u(ρ,θ)",
		xlabel="First n-Order",
		ylabel="Runtime (s)"
	)
	
	# 绘制时间图
	draw_runtime(
		sub_fig_enable=False,
		legend_enable=True,
		legend_pos=(0.17,0.15),
		draw_sel=3,
		csv_data_list=csv_data_list,
		label_list=label_list,
		xticks_list=xticks_list,
		yticks_list=yticks_list,
		fig_id=2,
		dpi=figure_dpi,
		save_path='./execution_runtime.jpg',
		title="Execution times of u(ρ,θ) with loading data",
		xlabel="First n-Order",
		ylabel="Runtime (s)"
	)
	
	# 画图
	#plt.show()