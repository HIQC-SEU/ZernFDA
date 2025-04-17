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
from zernipax_cpu import test_zernipax

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
def test_ours_cpu(
	info_list,
	filepath,
	max_n,
	list_mn,
	rho_list,
	theta_list,
	wgt_list
):
	# 打印
	print("> INFO: [python] Start ours evaluation on CPU platform!")
	
	# 设置设备为CPU
	device = torch.device("cpu")
	print("> INFO: [python] avaliable device: {0}.".format(device))
	
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
		pin_memory_enable=True
	)
	
	# 记录结束时间
	end_time = time.time()
	t_ours_cpu_load = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Ours Zernike-CPU finishes initialization! This step consumes {2} s".format(info_list[0],info_list[1],t_ours_cpu_load))
	
	# 记录起始时间
	start_time = time.time()
	
	# 前向传播
	outputs = zrnk_layer()
	
	# 记录结束时间
	end_time = time.time()
	t_ours_cpu = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Ours Zernike-CPU consumes {2} s".format(info_list[0],info_list[1],t_ours_cpu))
	
	# 保存测试时间
	save_test_time(
		order=info_list[0],
		time=info_list[1],
		filepath=filepath,
		str_name='ours-cpu',
		t_load=t_ours_cpu_load,
		t_cal=t_ours_cpu
	)
	
	# 返回
	return outputs.tolist(), t_ours_cpu, t_ours_cpu_load
	
# 测试Zern
def test_zern(
	info_list,
	filepath,
	list_mn,
	rho_list,
	theta_list,
	wgt_list
):
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Start zern evaluation on CPU platform!".format(info_list[0],info_list[1]))
	
	# 定义列表
	res_list = []
	
	# 记录起始时间
	start_time = time.time()
	
	# 初始化
	zern = Zernike(0)
	
	# 记录结束时间
	end_time = time.time()
	t_zern_load = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zernpy finishes initialization! This step consumes {2} s".format(info_list[0],info_list[1],t_zern_load))
	
	# 记录起始时间
	start_time = time.time()
	
	# 遍历
	for i in tqdm(range(0,len(rho_list)), mininterval=0.5):
		# 求和数据
		sum_data = 0.0
		
		# 遍历
		for j in range(0,len(list_mn)):
			# 解m,n
			m,n = list_mn[j]
			
			# 计算
			sum_data = sum_data + wgt_list[j] * zern.Z_nm(n=n, m=m, rho=rho_list[i], theta=theta_list[i], normalize_noll=False, mode='Jacobi')
			
		# 添加
		res_list.append(sum_data)
		
	# 记录结束时间
	end_time = time.time()
	t_zern = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zern consumes {2} s".format(info_list[0],info_list[1],t_zern))
	
	# 保存测试时间
	save_test_time(
		order=info_list[0],
		time=info_list[1],
		filepath=filepath,
		str_name='zern-cpu',
		t_load=t_zern_load,
		t_cal=t_zern
	)
	
	# 返回
	return res_list, t_zern, t_zern_load
	
# 测试zernike
def test_zernike(
	info_list,
	filepath,
	list_mn,
	rho_list,
	theta_list,
	wgt_list
):
	# 打印
	print("> INFO: [python] Start {0}th order, {1}th test time. zernike evaluation on CPU platform!".format(info_list[0],info_list[1]))
	
	# 定义列表
	res_list = []
	k_list = []
	
	# 记录起始时间
	start_time = time.time()
	
	# 初始化
	RZernike = RZern(len(list_mn))
	
	# 遍历
	for j in range(0,len(list_mn)):
		# 解m,n
		m,n = list_mn[j]
		
		# 获取NOLL标准的k,由于排除了0,所以最后要-1
		k = RZern.nm2noll(n=n, m=m)
		
		# 添加
		k_list.append(k - 1)
	
	# 记录结束时间
	end_time = time.time()
	t_zernike_load = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zernike finishes initialization! This step consumes {2} s".format(info_list[0],info_list[1],t_zernike_load))
	
	# 记录起始时间
	start_time = time.time()
	
	# 遍历
	for i in tqdm(range(0,len(rho_list)), mininterval=0.5):
		# 求和数据
		sum_data = 0.0
		
		# 遍历
		for j in range(0,len(k_list)):
			# 计算
			sum_data = sum_data + wgt_list[j] * RZernike.Zk(k=k_list[j], rho=rho_list[i], theta=theta_list[i])
		
		# 添加
		res_list.append(sum_data)
		
	# 记录结束时间
	end_time = time.time()
	t_zernike = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zernike consumes {2} s".format(info_list[0],info_list[1],t_zernike))
	
	# 保存测试时间
	save_test_time(
		order=info_list[0],
		time=info_list[1],
		filepath=filepath,
		str_name='zernike-cpu',
		t_load=t_zernike_load,
		t_cal=t_zernike
	)
	
	# 返回
	return res_list, t_zernike, t_zernike_load
	
# 测试
def test_zernpy(
	info_list,
	filepath,
	list_mn,
	rho_list,
	theta_list,
	wgt_list
):
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Start zernpy evaluation on CPU platform!".format(info_list[0],info_list[1]))
	
	# 定义列表
	polynomials_list = []
	res_list = []
	
	# 记录起始时间
	start_time = time.time()
	
	# 遍历
	for m, n in list_mn:
		# 生成计算
		zp = ZernPol(m=m, n=n)
		
		# 添加
		polynomials_list.append(zp)
	
	# 记录结束时间
	end_time = time.time()
	t_zernpy_load = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zernpy finishes initialization! This step consumes {2} s".format(info_list[0],info_list[1],t_zernpy_load))
	
	# 记录起始时间
	start_time = time.time()
	
	# 遍历
	for i in tqdm(range(0,len(rho_list)), mininterval=0.5):
		# 求和计算
		sum_data = ZernPol.sum_zernikes(
			coefficients=wgt_list, 
			polynomials=polynomials_list, 
			r=rho_list[i], 
			theta=theta_list[i]
		)
		
		# 添加进列表
		res_list.append(sum_data)
	
	# 记录结束时间
	end_time = time.time()
	t_zernpy = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zernpy consumes {2} s".format(info_list[0],info_list[1],t_zernpy))
	
	# 保存测试时间
	save_test_time(
		order=info_list[0],
		time=info_list[1],
		filepath=filepath,
		str_name='zernpy-cpu',
		t_load=t_zernpy_load,
		t_cal=t_zernpy
	)
	
	# 返回
	return res_list, t_zernpy, t_zernpy_load

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
	root_path = './evaluation/time/'
	
	# 检查生成目标路径
	target_path = check_path(root_path=root_path)
	
	# 目的文件名
	save_ours_cpu_time_filename = 'ours-cpu-time'
	save_zernipax_cpu_time_filename = 'zernipax-cpu-time'
	save_zernpy_cpu_time_filename = 'zernpy-cpu-time'
	save_zern_cpu_time_filename = 'zern-cpu-time'
	save_zernike_cpu_time_filename = 'zernike-cpu-time'
	
	# 文件路径
	save_ours_cpu_time_path = target_path + save_ours_cpu_time_filename + '.txt'
	save_zernipax_cpu_time_path = target_path + save_zernipax_cpu_time_filename + '.txt'
	save_zernpy_cpu_time_path = target_path + save_zernpy_cpu_time_filename + '.txt'
	save_zern_cpu_time_path = target_path + save_zern_cpu_time_filename + '.txt'
	save_zernike_cpu_time_path = target_path + save_zernike_cpu_time_filename + '.txt'
	
	# 生成文件内容头
	append_content_header_line(filepath=save_ours_cpu_time_path)
	append_content_header_line(filepath=save_zernipax_cpu_time_path)
	append_content_header_line(filepath=save_zernpy_cpu_time_path)
	append_content_header_line(filepath=save_zern_cpu_time_path)
	append_content_header_line(filepath=save_zernike_cpu_time_path)
	
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
			
			# 测试:Ours CPU
			ours_cpu_fig_list, t_ours_cpu, t_ours_cpu_load = test_ours_cpu(
				info_list=[polynomial_order,t],
				filepath=save_ours_cpu_time_path,
				max_n=max_n,
				list_mn=list_mn,
				rho_list=rho_list,
				theta_list=theta_list,
				wgt_list=wgt_list
			)
		
			# 测试:zernipax
			zernipax_fig_list, t_zernipax, t_zernipax_load = test_zernipax(
				info_list=[polynomial_order,t],
				filepath=save_zernipax_cpu_time_path,
				list_mn=list_mn,
				rho_list=rho_list,
				theta_list=theta_list,
				wgt_list=wgt_list
			)
			
			# 测试:zernpy
			zernpy_fig_list, t_zernpy, t_zernpy_load = test_zernpy(
				info_list=[polynomial_order,t],
				filepath=save_zernpy_cpu_time_path,
				list_mn=list_mn,
				rho_list=rho_list,
				theta_list=theta_list,
				wgt_list=wgt_list
			)
			
			# 测试:zern
			zern_fig_list, t_zern, t_zern_load = test_zern(
				info_list=[polynomial_order,t],
				filepath=save_zern_cpu_time_path,
				list_mn=list_mn,
				rho_list=rho_list,
				theta_list=theta_list,
				wgt_list=wgt_list
			)
			
			# 测试:zernike
			zernike_fig_list, t_zernike, t_zernike_load = test_zernike(
				info_list=[polynomial_order,t],
				filepath=save_zernike_cpu_time_path,
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