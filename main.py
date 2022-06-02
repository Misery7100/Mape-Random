import pandas as pd
import numpy as np
from random_mape import multi_get_output
from datetime import datetime
import argparse
import time
from progress.bar import IncrementalBar

class CustomBar(IncrementalBar):
	suffix = '%(index)d/%(max)d [%(el_tdm)s <- %(eta_tdm)s, %(get_avg).2farr/s]'

	@property
	def get_avg(self):
		return 1/self.avg

	@property
	def el_tdm(self):
		curr_el = str(self.elapsed_td).split(':')
		return f'{int(curr_el[0])*60 + int(curr_el[1]):02}:{int(curr_el[2]):02}'

	@property
	def eta_tdm(self):
		curr_el = str(self.eta_td).split(':')
		return f'{int(curr_el[0])*60 + int(curr_el[1]):02}:{int(curr_el[2]):02}'
	

def prepare_data(path):
	data_mape = pd.read_excel(path)
	return data_mape


def calculate_arrs(data, mape_want=0.2, idx=1, num=10):
	outout = None
	cnt = 0
	good = 0
	start = time.time()

	#bar = IncrementalBar('Generating output', max=num, suffix='%(index)d/%(max)d [%(elapsed_td)s < %(eta_td)s, %(avg).2fs/arr]')
	bar = CustomBar('Generating output', max=num)
	
	while good < num:
		if time.time() - start >= num*2: 
			break

		output = multi_get_output(data[data.ID == idx].drop('ID', axis=1).to_numpy()[0], mape_want=mape_want)
		if output: 
			arr, flag = output
			if flag == True: cnt += 1
			else:
				good += arr.shape[0]
				_ = [bar.next() for k in range(arr.shape[0])]
				if outout is None:
					outout = arr.reshape(arr.shape[0], -1)
				else:
					outout = np.concatenate([outout, arr.reshape(arr.shape[0], -1)], axis=0)
		else:
			cnt += 1
		if good >= num: break

	return outout


def save_excel(data, new_arrs, idx, dtnow, mape):
	new_file = pd.DataFrame(data=new_arrs).rename(columns=dict((i, v) for i, v in enumerate(data.columns[1:])))
	new_file.to_excel(f'id-{idx}_mape-{mape}_date_{dtnow}.xlsx', index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Stochastic estimations (mape)')

	parser.add_argument('--data', '-d', default='data_mape.xlsx',
						type=str, required=True,
						help='xlsx file with input data')

	parser.add_argument('--id', '-i', default=1,
						type=int, required=True,
						help='element index for estimations')

	parser.add_argument('--mape', '-m', default=0.2,
						type=float, required=True,
						help='desired mape for calculations')

	parser.add_argument('--num', '-n', default=10,
						type=int, required=True,
						help='desired number of output arrays')

	args = parser.parse_args()

	print(f'\nStarting calculations\n\nid\t\t{args.id}\nmape\t\t{args.mape}\noutputs\t\t{args.num}\n')
	print(f'Read data from: {args.data}\n')

	data = prepare_data(args.data)
	new_arrs = calculate_arrs(data=data, mape_want=args.mape, idx=args.id, num=args.num)

	time.sleep(0.5)
	dtnow = datetime.now().strftime('%d-%b-%Y_%Hh%Mm')
	print(f'\n\nSaved to: id-{args.id}_mape-{args.mape}_date_{dtnow}.xlsx')

	save_excel(data=data, new_arrs=new_arrs, idx=args.id, dtnow=dtnow, mape=args.mape)