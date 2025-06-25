import csv
import matplotlib.pyplot as plt
import numpy as np

# 绘制投毒攻击效果图片 0,0.2,0.4
mal_ratio = {'0.0', '0.2', '0.4'}
file_prefix = "output_acc_wo_detection_mal_"

plt.figure(figsize=(10, 5))
for ratio in mal_ratio:
	csv_reader = csv.reader(open(file_prefix + str(ratio) + '.csv'))
	for row in csv_reader:
		float_row = [float(x) for x in row]
		plt.plot(float_row, label=ratio)

plt.title("Accuracy without Detection")
plt.xlabel("Round")
plt.ylabel("Accuracy (%)")
plt.legend(title="Malicious Ratio")
plt.grid()
plt.show()

# 绘制恶意客户端检测效果对比图
mal_ratio = ['0.1', '0.2', '0.3', '0.4']
file_prefix_w = "output_acc_w_detection_mal_0."
file_prefix_wo = "output_acc_wo_detection_mal_0."

plt.figure(figsize=(10, 8))
for ratio in range(1, 5):
	csv_reader = csv.reader(open(file_prefix_w + str(ratio) + '.csv'))
	plt.subplot(220 + ratio)
	for row in csv_reader:
		float_row = [float(x) for x in row]
		plt.plot(float_row, label='With Detection')
	csv_reader = csv.reader(open(file_prefix_wo + str(ratio) + '.csv'))
	for row in csv_reader:
		float_row = [float(x) for x in row]
		plt.plot(float_row, label='Without Detection')

	plt.title("Accuracy, malicious ratio = " + str(ratio))
	plt.xlabel("Round")
	plt.ylabel("Accuracy (%)")
	plt.legend()
	plt.grid()

plt.tight_layout()
plt.show()