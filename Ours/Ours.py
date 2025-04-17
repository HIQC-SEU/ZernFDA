import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from datetime import datetime
from pathlib import Path

import os

# 添加文件内容的头行
def append_content_header_line(
	filepath
):
	# 打开文件
	with open(filepath,'a') as f_write:
		# 写入
		f_write.write('Order,Test-Times,Load-Time,Cal-Time,\n')
	
	# 打印
	print("> INFO: [python] The content header of the file '{0}' has been written successfully!".format(filepath))
	
# 检查路径
def check_path(
	root_path:str
)->Path:

	# 生成时间文件夹字符串
	time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

	# 构建完整路径
	target_path = Path(root_path) / time_str

	# 如果路径不存在,则创建
	if not os.path.exists(target_path):
		# 安全创建目录
		target_path.mkdir(parents=True, exist_ok=False)

	# 返回
	return str(target_path) + '/'
	
# 保存测试时间
def save_test_time(
	order,
	time,
	filepath,
	str_name,
	t_load,
	t_cal
):
	# 打开权重文件
	with open(filepath,'a') as f_write:
		f_write.write(str(order) + ',' + str(time) + ',' + str(t_load) + ',' + str(t_cal) + ',\n')

# 生成渐变色
def generate_gradient(
	start_color,
	end_color,
	steps
):
    """生成两色渐变"""
    start = np.array(start_color)
    end = np.array(end_color)
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    gradient = (1 - t) * start + t * end
    return np.clip(gradient, 0, 255).astype(np.uint8)

# 定制化彩虹色
def custom_rainbow_gradient(segments_steps):
	"""
	生成自定义多段彩虹色渐变
	:param segments_steps: 包含各段步数的列表，例如 [20, 40, 40, 40, 40, 20]
	"""
	# 定义颜色节点RGB值
	colors_list = [
		(0, 0, 0),    # 黑蓝色
		(0, 0, 60),   # 纯蓝色
		(0, 0, 255),   # 纯蓝色
		(0, 255, 255), # 亮蓝色
		(0, 240, 0),   # 纯绿色
		(255, 255, 0), # 纯黄色
		(255, 128, 0), # 橙色
		(255, 0, 0),   # 纯红色
		(50, 0, 0)    # 黑红色
	]
	
	# 确保长度正确
	assert len(segments_steps) == len(colors_list)-1, "步数列表长度应为颜色节点数减1"

	gradient_list = []

	# 遍历生成
	for i in range(len(colors_list)-1):
		# 生成两色渐变
		segment = generate_gradient(colors_list[i], colors_list[i+1], segments_steps[i])

		# 避免重复端点
		gradient_list.extend(segment[:-1])

		# 添加最后一个颜色
		gradient_list.append(colors_list[i+1])

	return np.array(gradient_list)

# 将计算生成的结果转化成RGB图像
def convert_image(
	fig_max,
	rho_list,
	acu_list,
	gradient_color_list,
	image_size=[1080,1920]
):
	# 将输入数据预转换为numpy数组
	np_rho_list = np.array(rho_list, dtype=np.float32)
	np_acu_list = np.array(acu_list, dtype=np.float32)
	
	# 并行计算归一化系数
	norm_acu_list = np.where(np_rho_list == 0.0,0.0,(np_acu_list + fig_max)/(2 * fig_max))

	# 并行化颜色索引
	vec_indices = (norm_acu_list * 255).astype(np.uint16)  # 防止溢出
	vec_safe_indices = np.clip(vec_indices, 0, len(gradient_color_list)-1)  # 安全索引

	# 批量获取颜色数据
	image_list = gradient_color_list[vec_safe_indices]
	
	# 转换
	np_data = np.array(image_list, dtype=np.uint8).reshape(image_size[0], image_size[1], 3)

	# 直接重塑为目标尺寸 (避免列表操作)
	return Image.fromarray(np_data, 'RGB')

# 画图
def draw_fig(
	fig_list,
	rho_list,
	size_x=1920,
	size_y=1080
):
	# 记录起始时间
	start_time = time.time()
	
	# 记录最大值
	fig_max = 0.0
	
	# 生成器表达式+内置max
	fig_max = max(map(lambda x: abs(x), fig_list))
	
	# 定义每个颜色段的过渡速度
	segments_steps = [30, 40, 20, 16, 64, 30, 30, 26]

	# 获取渐变色
	gradient_color_list = custom_rainbow_gradient(segments_steps)
	
	# 将计算生成的结果转化成RGB图像
	img = convert_image(
		fig_max=fig_max,
		rho_list=rho_list,
		acu_list=fig_list,
		gradient_color_list=gradient_color_list,
		image_size=[size_y,size_x]
	)
	
	# 记录结束时间
	end_time = time.time()
	t_draw = end_time - start_time
	
	# 打印
	print("> INFO: [python] draw figure consumes {0} s".format(t_draw))
	
	# 显示
	plt.figure(figsize=(12, 6))
	plt.imshow(img)
	plt.axis('off')
	plt.title("RGB Image (Matplotlib)")
	plt.show()

# 根据多项式阶数生成s
def get_s_from_order(
	polynomial_order
):
	# 定义s
	s = 0
	
	# 遍历
	for n in range(0,polynomial_order + 1):
		for m in range(-n,n + 1):
			# 如果n-m为奇数,跳过
			if (n-m) % 2 == 1:
				continue
			
			# 计数
			s = s + 1
	
	# 返回
	return s
	
# 生成前s项需要的(m,n)列表
def generate_s_terms_list(
	s
):
	# 定义结果列表
	list_nm = []
	list_mn = []
	
	# 计数
	gen_cnt = 0
	
	# 记录最大的n和最大的m
	max_n = 0
	max_m = 0
	min_m = 0
	
	# 遍历
	for n in range(s):
		for m in range(-n,n + 1):
			# 如果n-m为奇数,跳过
			if (n-m) % 2 == 1:
				continue
			
			# 添加
			list_nm.append([n,m])
			
			# 判断添加
			max_n = max(n,max_n)
			max_m = max(m,max_m)
			min_m = min(m,min_m)
			
			# 计数
			gen_cnt = gen_cnt + 1
			
			# 如果计数超过
			if gen_cnt > s - 1:
				# 中断
				break
		
		# 如果计数超过
		if gen_cnt > s - 1:
			# 清零
			gen_cnt = 0
			
			# 中断
			break
	
	# 遍历
	for m in range(min_m,max_m+1):
		for n in range(abs(m),max_n+1):
			# 如果n-m为奇数,跳过
			if (n-m) % 2 == 1:
				continue
			
			# 判断是否存在
			if [n,m] not in list_nm:
				continue
				
			# 添加
			list_mn.append([m,n])
			
			# 计数
			gen_cnt = gen_cnt + 1
			
			# 如果计数超过
			if gen_cnt > s - 1:
				# 中断
				break
		
		# 如果计数超过
		if gen_cnt > s - 1:
			# 清零
			gen_cnt = 0
			
			# 中断
			break
	
	# 返回
	return list_nm, list_mn

# 产生径向多项式系数
def get_radial_polynomial_coefficients(
	n,
	m
):

	# 如果n-m为奇数,直接返回0
	if (n-m) % 2 == 1:
		return 0
	
	# 如果m=0
	if m == 0:
		return n + 1
	else:
		return 2 * (n + 1)

# 生成前s项需要的(m,n)列表
def generate_s_terms_list(
	s
):
	# 定义结果列表
	list_nm = []
	list_mn = []
	
	# 计数
	gen_cnt = 0
	
	# 记录最大的n和最大的m
	max_n = 0
	max_m = 0
	min_m = 0
	
	# 遍历
	for n in range(s):
		for m in range(-n,n + 1):
			# 如果n-m为奇数,跳过
			if (n-m) % 2 == 1:
				continue
			
			# 添加
			list_nm.append([n,m])
			
			# 判断添加
			max_n = max(n,max_n)
			max_m = max(m,max_m)
			min_m = min(m,min_m)
			
			# 计数
			gen_cnt = gen_cnt + 1
			
			# 如果计数超过
			if gen_cnt > s - 1:
				# 中断
				break
		
		# 如果计数超过
		if gen_cnt > s - 1:
			# 清零
			gen_cnt = 0
			
			# 中断
			break
	
	# 遍历
	for m in range(min_m,max_m+1):
		for n in range(abs(m),max_n+1):
			# 如果n-m为奇数,跳过
			if (n-m) % 2 == 1:
				continue
			
			# 判断是否存在
			if [n,m] not in list_nm:
				continue
				
			# 添加
			list_mn.append([m,n])
			
			# 计数
			gen_cnt = gen_cnt + 1
			
			# 如果计数超过
			if gen_cnt > s - 1:
				# 中断
				break
		
		# 如果计数超过
		if gen_cnt > s - 1:
			# 清零
			gen_cnt = 0
			
			# 中断
			break
	
	# 返回
	return list_nm, list_mn, max_n + 1

# 产生权重
def get_weight_list(
	list_mn
):
	# 定义权重列表
	wgt_list = []
	
	# 遍历
	for m, n in list_mn:
		# 如果m
		if m == 1 and n == 5:
			wgt_list.append(0.5)
		elif m == -1 and n == 5:
			wgt_list.append(0.5)
		else:
			wgt_list.append(0.0)
	
	# 返回
	return wgt_list

# 生成坐标
def get_rho_coord(
	rho_max=540,
	x_range=[1,1921],
	y_range=[1,1081],
	center_xy=[960,540]
):
	# 定义列表
	rho_list = []
	theta_list = []
	
	# 解析中点
	x0, y0 = center_xy
	
	# 提前计算好
	for y in range(y_range[0],y_range[1]):
		for x in range(x_range[0],x_range[1]):
			# 计算横纵坐标差值
			sqrt_delta_x = x - x0
			sqrt_delta_y = y - y0
			
			# 计算横纵坐标差值平方
			sqrt_square_x = sqrt_delta_x * sqrt_delta_x
			sqrt_square_y = sqrt_delta_y * sqrt_delta_y
			
			# 计算平方根
			sqrt_xy = np.sqrt(sqrt_square_x + sqrt_square_y)
			
			# 计算角度值
			theta = math.atan2(sqrt_delta_y,sqrt_delta_x)
			
			# 约束角度
			if theta > 2 * math.pi:
				theta = theta - 2 * math.pi
			elif theta < 0:
				theta = theta + 2 * math.pi
				
			# 如果坐标超过最大值
			if sqrt_xy > rho_max:
				normalized_rho = 0.0
			else:
				# 计算归一化数值
				normalized_rho = sqrt_xy / rho_max
			
			# 生成参数
			rho_list.append(normalized_rho)
			theta_list.append(theta)
	
	# 返回
	return rho_list, theta_list

# 生成径向多项式
def generate_radial_polynomial(
	max_n,
	polynomial_coefficient,
	n,
	m
):
	# 定义结果列表
	res = [0.0] * (max_n + 1)
	
	# 如果n-m为奇数,直接返回
	if (n-m) % 2 == 1:
		return res
	
	# 计算
	mid_nm = int((n + abs(m))/2)
	delta_nm = int((n - abs(m))/2)
	
	# 否则,遍历
	for k in range(0,delta_nm + 1):
		
		# 计算阶乘
		k_factorial = math.factorial(k)
		mid_factorial = math.factorial(mid_nm - k)
		delta_factorial = math.factorial(delta_nm - k)
		delta_nk_factorial = math.factorial(n - k)
		
		# 计算系数
		fac_rho = pow(-1,k) * delta_nk_factorial
		fac_div = (k_factorial * mid_factorial * delta_factorial)
		fac_rho = fac_rho / fac_div
		
		# 计算幂次
		rho_index = n - 2 * k
		
		# 更新
		res[rho_index] = polynomial_coefficient * fac_rho
	
	# 返回
	return res

# 编译权重
def compile_bjm(
	s,
	wgt_list,
	max_n
):
	# 定义计数
	gen_cnt = 0
	
	# 定义列表
	bjm_list = []
	
	# 加法
	def add_lists(list1, list2):
		return [x + y for x, y in zip(list1, list2)]
	
	# 遍历
	for m in range(-max_n,max_n + 1):
		# 定义求和
		partial_sum = [0.0] * (max_n + 1)
		
		# 遍历n
		for n in range(abs(m),max_n + 1,2):
			
			# 生成多项式系数
			polynomial_coefficient = get_radial_polynomial_coefficients(
				n=n,
				m=m
			)
			
			# 生成径向多项式
			radial_ploynomial = generate_radial_polynomial(
				max_n=max_n,
				polynomial_coefficient=wgt_list[gen_cnt] * np.sqrt(polynomial_coefficient),
				n=n,
				m=m
			)
			
			# 求和
			partial_sum = add_lists(partial_sum,radial_ploynomial)
			
			# 计数
			gen_cnt = gen_cnt + 1
			
			# 如果计数超过
			if gen_cnt > s - 1:
				# 中断
				break
		
		# 保存
		bjm_list.append(partial_sum)
		
		# 如果计数超过
		if gen_cnt > s - 1:
			# 清零
			gen_cnt = 0
			
			# 中断
			break
	
	# 返回
	return bjm_list

# 编译A
def compile_A(
	max_n,
	theta_list
):
	# 预转换避免重复类型转换
	theta_arr = np.array(theta_list)
	
	# 定义
	A_list = []

	# 生成负数部分(m < 0)
	for m in range(max_n, 0,-1):
		A_list.append(np.sin(m * theta_arr).tolist())

	# 生成非负部分(m >= 0)
	for m in range(0, max_n+1):
		A_list.append(np.cos(m * theta_arr).tolist())

	# 返回
	return A_list

# Zernike类
class Zernike_Layer(torch.nn.Module):
	# 初始化
	def __init__(
		self, 
		max_n,
		list_mn,
		rho_list,
		theta_list,
		wgt_list,
		device,
		pin_memory_enable=True
	):
		# 继承类初始化
		super().__init__()

		# 定义最大的n
		self.max_n = max_n
		self.max_s = len(list_mn)
		self.full_size = len(rho_list)
		
		# 转化成张量列表
		self.rho_tensor_list = torch.tensor(rho_list,dtype=torch.float32)
		
		# 编译bjm
		bjm_list = compile_bjm(
			s=self.max_s,
			wgt_list=wgt_list,
			max_n=self.max_n-1
		)
		
		# 编译A
		acu_A_list = compile_A(
			max_n=self.max_n-1,
			theta_list=theta_list
		)

		# 径向多项式张量
		self.rcu_tensor = torch.ones(
			self.full_size, 
			self.max_n, 
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
		# 径向多项式参数张量
		self.rcu_bjm_tensor = torch.tensor(
			bjm_list,
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
		# 角项函数张量
		self.acu_tensor = torch.zeros(
			self.full_size, 
			self.max_s, 
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
		# 角度张量
		self.acu_A_tensor = torch.tensor(
			acu_A_list,
			device=device,
			pin_memory=pin_memory_enable, 
			dtype=torch.float32
		)
		
		# 结果图片张量
		self.fig_tensor = torch.zeros(
			self.full_size, 
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
	# 前向
	def forward(self):
		
		# 遍历计算幂次
		cum_prod = torch.cumprod(self.rho_tensor_list.unsqueeze(1).expand(-1, self.max_n), dim=1)
		self.rcu_tensor[:, 1:self.max_n] = cum_prod[:, :-1]

		# 计算矩阵乘,转置为(N, max_s)
		batch_mat = torch.matmul(
			self.rcu_tensor, 
			self.rcu_bjm_tensor.T  
		)
		
		# 结果形状 (M, max_s)
		self.acu_tensor = batch_mat * self.acu_A_tensor.T

		# 计算图片
		self.fig_tensor = self.acu_tensor.sum(dim=1)
		
		# 返回
		return self.fig_tensor

# Zernike类
class Zernike_Layer_WithNoOptimization(torch.nn.Module):
	# 初始化
	def __init__(
		self, 
		max_n,
		list_mn,
		rho_list,
		theta_list,
		wgt_list,
		device,
		pin_memory_enable=True
	):
		# 继承类初始化
		super().__init__()

		# 定义最大的n
		self.max_n = max_n
		self.max_s = len(list_mn)
		self.full_size = len(rho_list)
		
		# 转化成张量列表
		self.rho_tensor_list = torch.tensor(rho_list,dtype=torch.float32)
		
		# 定义列表
		rcu_coefficients_list = []
		acu_theta_list = []
		
		# 转换成
		theta_arr = np.array(theta_list)
		
		# 遍历m,n
		for i in tqdm(range(len(list_mn))):
			# 解析
			m, n = list_mn[i]
			
			# 生成多项式系数
			polynomial_coefficient = get_radial_polynomial_coefficients(
				n=n,
				m=m
			)
			
			# 生成径向多项式
			radial_ploynomial = generate_radial_polynomial(
				max_n=self.max_n - 1,
				polynomial_coefficient=wgt_list[i] * np.sqrt(polynomial_coefficient),
				n=n,
				m=m
			)
			
			# 添加
			rcu_coefficients_list.append(radial_ploynomial)
			
			# 生成sin和cos
			if m >= 0:
				acu_theta_list.append(np.cos(m * theta_arr).tolist())
			else:
				acu_theta_list.append(np.sin(abs(m) * theta_arr).tolist())

		# 径向多项式张量
		self.rcu_tensor = torch.ones(
			self.full_size, 
			self.max_n, 
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
		# 径向多项式参数张量
		self.rcu_coefficients_tensor = torch.tensor(
			rcu_coefficients_list,
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
		# 角项函数张量
		self.acu_tensor = torch.zeros(
			self.full_size, 
			self.max_s, 
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
		# 角度张量
		self.acu_theta_tensor = torch.tensor(
			acu_theta_list,
			device=device,
			pin_memory=pin_memory_enable, 
			dtype=torch.float32
		)
		
		# 结果图片张量
		self.fig_tensor = torch.zeros(
			self.full_size, 
			device=device,
			pin_memory=pin_memory_enable,
			dtype=torch.float32
		)
		
	# 前向
	def forward(self):
		
		# 遍历计算幂次
		cum_prod = torch.cumprod(self.rho_tensor_list.unsqueeze(1).expand(-1, self.max_n), dim=1)
		self.rcu_tensor[:, 1:self.max_n] = cum_prod[:, :-1]

		# 计算矩阵乘,转置为(N, max_s)
		batch_mat = torch.matmul(
			self.rcu_tensor, 
			self.rcu_coefficients_tensor.T  
		)
		
		# 结果形状 (M, max_s)
		self.acu_tensor = batch_mat * self.acu_theta_tensor.T

		# 计算图片
		self.fig_tensor = self.acu_tensor.sum(dim=1)
		
		# 返回
		return self.fig_tensor

# 测试无优化版本
def test_zrnk_layer_nopt(
	max_n,
	list_mn,
	rho_list,
	theta_list,
	wgt_list,
	device,
	pin_memory_enable
):

	# 记录起始时间
	start_time = time.time()
	
	# 无优化的测试
	zrnk_layer_nopt = Zernike_Layer_WithNoOptimization(
		max_n=max_n,
		list_mn=list_mn,
		rho_list=rho_list,
		theta_list=theta_list,
		wgt_list=wgt_list,
		device=device,
		pin_memory_enable=pin_memory_enable,
	)
	
	# 记录结束时间
	end_time = time.time()
	
	# 打印
	print("> INFO: [python] Zernike with no optimization finishes initialization! This step consumes {0} s".format((end_time - start_time)))
	
	# 记录起始时间
	start_time = time.time()
	
	# 前向传播
	zrnk_layer_nopt_outputs = zrnk_layer_nopt()
	
	# 记录结束时间
	end_time = time.time()
	
	print("> INFO: [python] Zernike with no optimization zernike runtime is {0} s".format((end_time - start_time)))
	
	# 返回
	return zrnk_layer_nopt_outputs.to(torch.device("cpu")).tolist()

# 测试优化版本
def test_zrnk_layer(
	max_n,
	list_mn,
	rho_list,
	theta_list,
	wgt_list,
	device,
	pin_memory_enable
):
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
		pin_memory_enable=pin_memory_enable,
	)
	
	# 记录结束时间
	end_time = time.time()
	
	# 打印
	print("> INFO: [python] Proposed zernike finishes initialization! This step consumes {0} s".format((end_time - start_time)))
	
	# 记录起始时间
	start_time = time.time()
	
	# 前向传播
	zrnk_layer_outputs = zrnk_layer()
	
	# 记录结束时间
	end_time = time.time()
	
	print("> INFO: [python] Proposed zernike runtime is {0} s".format((end_time - start_time)))
	
	# 返回
	return zrnk_layer_outputs.to(torch.device("cpu")).tolist()
	
# 主函数
if __name__ == "__main__":
	
	# 在初始化代码中添加
	torch.cuda.set_per_process_memory_fraction(0.8)  # 防止OOM
	torch.cuda.empty_cache()  # 清理碎片
	
	# 设置参数
	show_enable = True		# 显示使能
	use_gpu = True			# 是否使用GPU
	pin_memory_enable=True	# 是否启用内存锁定使能
	
	# 是否使用GPU
	if use_gpu == True and torch.cuda.is_available():
		device = torch.device("cuda:0")
		pin_memory_enable = False	# GPU使用时,禁止该使能
		print("> INFO: [python] avaliable device: {0}. Current cuda device num is {1}".format(device,torch.cuda.device_count()))
	else:
		device = torch.device("cpu")
		print("> INFO: [python] avaliable device: {0}.".format(device))
	
	# 定义坐标参数
	rho_max = 540
	x_range = [1,1921]
	y_range = [1,1081]
	center_xy = [960,540]
	
	# 生成s
	s=get_s_from_order(
		polynomial_order=14
	)
			
	# 生成前s项需要的
	list_nm, list_mn, max_n = generate_s_terms_list(
		s=s
	)
	
	# 生成权重
	wgt_list = get_weight_list(
		list_mn=list_mn
	)
	
	# 生成极坐标
	rho_list, theta_list = get_rho_coord(
		rho_max=rho_max,
		x_range=x_range,
		y_range=y_range,
		center_xy=center_xy
	)
	
	# 测试无优化版本
	zrnk_nopt_fig_list = test_zrnk_layer_nopt(
		max_n=max_n,
		list_mn=list_mn,
		rho_list=rho_list,
		theta_list=theta_list,
		wgt_list=wgt_list,
		device=device,
		pin_memory_enable=pin_memory_enable,
	)
	
	# 测试优化版本
	zrnk_fig_list = test_zrnk_layer(
		max_n=max_n,
		list_mn=list_mn,
		rho_list=rho_list,
		theta_list=theta_list,
		wgt_list=wgt_list,
		device=device,
		pin_memory_enable=pin_memory_enable,
	)
	
	# 如果需要显示图片
	if show_enable == True:
		# 画图
		draw_fig(
			fig_list=zrnk_fig_list,
			rho_list=rho_list,
			size_x=x_range[1] - 1,
			size_y=y_range[1] - 1
		)