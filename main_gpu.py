import sys
import os

# 基础库
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
import mpmath
import timeit
from tqdm import tqdm
import math
import time

import torch

# 添加路径
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("./Ours/"))
sys.path.append(os.path.abspath("./ZERN/"))
sys.path.append(os.path.abspath("./zernike/"))
sys.path.append(os.path.abspath("./zernipax/"))
sys.path.append(os.path.abspath("./zernpy/src/"))

# 添加测试库
from zern.zern_core import Zernike
from zernike import RZern
from zernpy import ZernPSF, ZernPol
from zernipax_gpu import test_zernipax

import jax.numpy as jnp

from Ours import Zernike_Layer, generate_gradient, custom_rainbow_gradient, convert_image, generate_s_terms_list, get_rho_coord, draw_fig, get_s_from_order, save_test_time, check_path, append_content_header_line

# 产生权重
def get_test_weight_list(
	list_mn
):
	# 定义权重列表
	wgt_list = np.random.normal(0.5, 0.6, len(list_mn))
	
	# 返回
	return wgt_list

# 测试我们的
def test_ours_gpu(
	info_list,
	filepath,
	max_n,
	list_mn,
	rho_list,
	theta_list,
	wgt_list
):
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Start ours reproduction evaluation on GPU platform!".format(info_list[0],info_list[1]))
	
	# 在初始化代码中添加
	torch.cuda.set_per_process_memory_fraction(0.8)  # 防止OOM
	torch.cuda.empty_cache()  # 清理碎片
	
	# 设置设备为GPU
	device = torch.device("cuda:0")
	print("> INFO: [python] avaliable device: {0}. Current cuda device num is {1}".format(device,torch.cuda.device_count()))
	
	# 记录起始时间
	start_time = time.time()
	
	# 创建径向多项式层
	zrnk_layer = Zernike_Layer(
		max_n=max_n,
		list_mn=list_mn,
		rho_list=rho_list,
		theta_list=theta_list,
		wgt_list=wgt_list,
		device=device,
		pin_memory_enable=False
	)
	
	# 记录结束时间
	end_time = time.time()
	t_ours_gpu_load = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Ours Zernike-GPU finishes initialization! This step consumes {2} s".format(info_list[0],info_list[1],t_ours_gpu_load))
	
	# 记录起始时间
	start_time = time.time()
	
	# 前向传播
	outputs = zrnk_layer()
	
	# 记录结束时间
	end_time = time.time()
	t_ours_gpu = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Ours Zernike-GPU consumes {2} s".format(info_list[0],info_list[1],t_ours_gpu))
	
	# 保存测试时间
	save_test_time(
		order=info_list[0],
		time=info_list[1],
		filepath=filepath,
		str_name='ours-gpu',
		t_load=t_ours_gpu_load,
		t_cal=t_ours_gpu
	)
	
	# 返回
	return outputs.to(torch.device("cpu")).tolist(), t_ours_gpu, t_ours_gpu_load

# 主函数
if __name__ == "__main__":
	
	# 画图
	fig_sel = None
	
	# 定义测试次数
	test_len = 1
	
	# 定义测试阶数
	polynomial_order_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	
	# 定义坐标参数
	rho_max = 540
	x_range = [1,1921]
	y_range = [1,1081]
	center_xy = [960,540]
	
	# 根目录
	root_path = './evaluation/time'
	
	# 检查生成目标路径
	target_path = check_path(root_path=root_path)
	
	# 目的文件名
	save_ours_gpu_time_filename = 'ours-gpu-time'
	save_zernipax_gpu_time_filename = 'zernipax-gpu-time'
	
	# 文件路径
	save_ours_gpu_time_path = target_path + save_ours_gpu_time_filename + '.txt'
	save_zernipax_gpu_time_path = target_path + save_zernipax_gpu_time_filename + '.txt'
	
	# 生成文件内容头
	append_content_header_line(filepath=save_ours_gpu_time_path)
	append_content_header_line(filepath=save_zernipax_gpu_time_path)
	
	# 生成极坐标
	rho_list, theta_list = get_rho_coord(
		rho_max=rho_max,
		x_range=x_range,
		y_range=y_range,
		center_xy=center_xy
	)
			
	# 遍历阶数
	for polynomial_order in polynomial_order_list:
		# 遍历进行测试
		for t in range(0, test_len):
			# 生成s
			s=get_s_from_order(
				polynomial_order=polynomial_order
			)
			
			# 生成前s项需要的
			list_nm, list_mn, max_n = generate_s_terms_list(
				s=s
			)
			
			# 生成权重
			wgt_list = get_test_weight_list(
				list_mn=list_mn
			)
			
			# 测试:Ours GPU
			ours_gpu_fig_list, t_ours_gpu, t_ours_gpu_load = test_ours_gpu(
				info_list=[polynomial_order,t],
				filepath=save_ours_gpu_time_path,
				max_n=max_n,
				list_mn=list_mn,
				rho_list=rho_list,
				theta_list=theta_list,
				wgt_list=wgt_list
			)
			
			# 测试:zernipax
			zernipax_fig_list, t_zernipax, t_zernipax_load = test_zernipax(
				info_list=[polynomial_order,t],
				filepath=save_zernipax_gpu_time_path,
				list_mn=list_mn,
				rho_list=rho_list,
				theta_list=theta_list,
				wgt_list=wgt_list
			)
	
	# 判断画图
	if fig_sel == "ours-gpu":
		# 画图
		draw_fig(
			fig_list=ours_gpu_fig_list,
			rho_list=rho_list,
			size_x=x_range[1] - 1,
			size_y=y_range[1] - 1
		)
	elif fig_sel == "ours-cpu":
		# 画图
		draw_fig(
			fig_list=ours_cpu_fig_list,
			rho_list=rho_list,
			size_x=x_range[1] - 1,
			size_y=y_range[1] - 1
		)
	elif fig_sel == "zernipax":
		# 画图
		draw_fig(
			fig_list=zernipax_fig_list,
			rho_list=rho_list,
			size_x=x_range[1] - 1,
			size_y=y_range[1] - 1
		)
	elif fig_sel == "zernpy":
		# 画图
		draw_fig(
			fig_list=zernpy_fig_list,
			rho_list=rho_list,
			size_x=x_range[1] - 1,
			size_y=y_range[1] - 1
		)
	
	elif fig_sel == "zern":
		# 画图
		draw_fig(
			fig_list=zern_fig_list,
			rho_list=rho_list,
			size_x=x_range[1] - 1,
			size_y=y_range[1] - 1
		)
	
	elif fig_sel == "zernike":
		# 画图
		draw_fig(
			fig_list=zernike_fig_list,
			rho_list=rho_list,
			size_x=x_range[1] - 1,
			size_y=y_range[1] - 1
		)