import sys
import os

# 基础库
import numpy as np
import time

# 添加路径
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("./Ours/"))
sys.path.append(os.path.abspath("./zernipax/"))

# 添加测试库
from zernipax import set_device
set_device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 防止日志打印
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from zernipax.basis import ZernikePolynomial
from zernipax.zernike import *
from zernipax.plotting import plot_mode
from zernipax.backend import jax, use_jax

import jax.numpy as jnp

from Ours import save_test_time


# 测试zernipax
def test_zernipax(
	info_list,
	filepath,
	list_mn,
	rho_list,
	theta_list,
	wgt_list
):
	# 设置CPU
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. Start zernipax evaluation on {2} platform!".format(info_list[0],info_list[1],"cpu"))
	
	# 记录起始时间
	start_time = time.time()
	
	# 转换为JAX数组
	jnp_list_mn = jnp.array(list_mn)
	jnp_rho_list = jnp.array(rho_list)
	jnp_theta_list = jnp.array(theta_list)
	jnp_wgt_list = jnp.array(wgt_list)
	wgt_len_list = jnp.arange(0,len(list_mn))
	
	# 获取m列表
	m_list = jnp_list_mn[:, 0]	
	
	# 预计算绝对值|m|
	abs_m = jnp.abs(m_list)
	
	# 生成角度向量
	theta_vec = jnp_theta_list.reshape(1, -1)
	
	# 记录结束时间
	end_time = time.time()
	t_zernipax_load = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zernipax-{2} load tensor consumes {3} s".format(info_list[0],info_list[1],"cpu",t_zernipax_load))
	
	# 记录起始时间
	start_time = time.time()
	
	# 提前计算三角函数
	sin_theta_list = jnp.sin(abs_m[:, None] * theta_vec)
	cos_theta_list = jnp.cos(m_list[:, None] * theta_vec)
	
	# 生成3态向量:
	trig_vec = jnp.where(
		m_list[:, None] < 0, 
		sin_theta_list, 
		jnp.where(m_list[:, None] == 0, 1.0, cos_theta_list)
	)
	
	# 计算1轮多项式
	def compute_radials(m,n):
		return jax.vmap(lambda rho: zernike_radial(rho, n, m)[0][0])(jnp_rho_list)

	# 计算多项式
	radial_vec = jax.vmap(lambda m, n, i: jnp_wgt_list[i] * compute_radials(m,n))(jnp_list_mn[:,0], jnp_list_mn[:,1],wgt_len_list[:])

	# 组合所有项并求和,形状广播自动完成
	def compute_all():
		return jnp.sum(radial_vec * trig_vec, axis=0)
	
	# 计算
	res_list = compute_all().block_until_ready()
	
	# 记录结束时间
	end_time = time.time()
	t_zernipax = end_time - start_time
	
	# 打印
	print("> INFO: [python] {0}th order, {1}th test time. zernipax-{2} consumes {3} s".format(info_list[0],info_list[1],"cpu",t_zernipax))
	
	# 保存测试时间
	save_test_time(
		order=info_list[0],
		time=info_list[1],
		filepath=filepath,
		str_name='zernipax-cpu',
		t_load=t_zernipax_load,
		t_cal=t_zernipax
	)
	
	# 返回
	return res_list.tolist(), t_zernipax, t_zernipax_load
	