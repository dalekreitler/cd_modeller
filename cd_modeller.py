import sys
import csv
import numpy
import re
import time
import pickle
from matplotlib import pylab
from glob import glob



class cell:

###################################################
#************ 2 state folder **********************
###################################################
#
#  F <=> U		
#  K = [U]/[F]
#  dG = dH*(1 - (1/Tm)) + 
#  dCp*(T - Tm - dCp*ln(T/Tm)		
#  alpha = 1 / (1 + K)	fraction folded
#  exp_ellipticity = alpha*folded_ellipticity +
#		   (1 - alpha)*unfolded_ellipticity
#
###################################################
###################################################

	def __init__(self, 
		     name="cell0.txt", 
		     dH=25, 
		     Tm=45, 
		     dCp=0, 
		     nf=5, 
		     nu=5, 
		     baseline="points", 
		     data_type="aviv"):

		self.baseline = baseline		
		self.data_type = data_type		
		self.name = name
		self.Tm = Tm + 273.15 #convert to Kelvin
		self.dH = dH
		self.nf = nf
		self.nu = nu
		self.dCp = dCp
		self.data = numpy.genfromtxt(self.name,delimiter=",")
		self.cd_signal = self.data[:,1]
		self.R = 1.987e-3		

		#aviv .dat files store information as
		#wavelen

		if self.data_type == "aviv":

			self.temp = self.data[:,4]
			self.error = self.data[:,2] 
		else:

			self.temp = self.data[:,0] 
			self.error = numpy.ones(len(self.temp))

		self.baselines()

		return

	def baselines(self):

		temp_f = self.temp[0:self.nf:1] + 273.15
		temp_uf = self.temp[len(self.temp)-self.nu:] + 273.15 
		cd_f = self.cd_signal[0:self.nf:1]
		cd_uf = self.cd_signal[len(self.temp) - self.nu:]	
		A_f = numpy.vstack([temp_f, numpy.ones(len(temp_f))]).T
		A_uf = numpy.vstack([temp_uf, numpy.ones(len(temp_uf))]).T
		self.m_f, self.y_int_f = numpy.linalg.lstsq(A_f,cd_f)[0]
		self.m_uf, self.y_int_uf = numpy.linalg.lstsq(A_uf,cd_uf)[0]

		return 

	def cd_calc(self):
	
		T = self.temp + 273.15		
		dG = (self.dH*(1-T / self.Tm) + 
		      self.dCp*(T - self.Tm - 
		      self.temp*numpy.log(T/self.Tm)))
		K = numpy.exp(-dG / (self.R * T))
		frac_fold = 1/(1 + K)
		cd_f = self.m_f*T + self.y_int_f
		cd_uf = self.m_uf*T + self.y_int_uf 
		cd_calc = cd_f*frac_fold + (1 - frac_fold)*cd_uf

		return cd_calc

	def frac_calc(self):

		T = self.temp + 273.15		
		dG = (self.dH*(1-T / self.Tm) + 
		      self.dCp*(T - self.Tm - 
		      self.temp*numpy.log(T/self.Tm)))
		K = numpy.exp(-dG / (self.R * T))
		frac_calc = 1/(1 + K)

		return frac_calc

	def jparams(self):

		dH = self.dH
		Tm = self.Tm
		dCp = self.dCp
		R = self.R
		T = self.temp + 273.15
		m_f = self.m_f
		y_int_f = self.y_int_f
		m_uf = self.m_uf
		y_int_uf = self.y_int_uf

		dfdH = ((numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
			T*numpy.log(T/Tm)))/(R*T))*(T/Tm - 1)*(y_int_uf + 
			T*m_uf))/(R*T*(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm -
		        T + T*numpy.log(T/Tm)))/(R*T)) + 1)**2) - 
			(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
			T*numpy.log(T/Tm)))/(R*T))*(T/Tm - 1)*(y_int_f + 
			T*m_f))/(R*T*(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
			T*numpy.log(T/Tm)))/(R*T)) + 1)**2))

		dfdTm = ((numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
			T*numpy.log(T/Tm)))/(R*T))*(dCp*(T/Tm - 1) + 
			(T*dH)/Tm**2)*(y_int_f + T*m_f))/
			(R*T*(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T +
			T*numpy.log(T/Tm)))/(R*T)) + 1)**2) - 
			(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
			T*numpy.log(T/Tm)))/(R*T))*(dCp*(T/Tm - 1) + 
			(T*dH)/Tm**2)*(y_int_uf + T*m_uf))/
			(R*T*(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
			T*numpy.log(T/Tm)))/(R*T)) + 1)**2))

		'''dfdCp = numpy.zeros(len(self.temp))
		dfdm_f = numpy.zeros(len(self.temp))
		dfdm_uf = numpy.zeros(len(self.temp))
		dfdy_int_f = numpy.zeros(len(self.temp))
		dfdy_int_uf = numpy.zeros(len(self.temp))'''
	
		if self.dCp != 0:

			dfdCp = ((numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
				T*numpy.log(T/Tm)))/(R*T))*(y_int_uf + 
				T*m_uf)*(Tm - T + T*numpy.log(T/Tm)))/
				(R*T*(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
				T*numpy.log(T/Tm)))/(R*T)) + 1)**2) - 
				(numpy.exp((dH*(T/Tm - 1) + dCp*(Tm - T + 
				T*numpy.log(T/Tm)))/(R*T))*(y_int_f + T*m_f)*(Tm - 
				T + T*numpy.log(T/Tm)))/(R*T*(numpy.exp((dH*(T/Tm - 1) + 
				dCp*(Tm - T + T*numpy.log(T/Tm)))/(R*T)) + 1)**2))

			jparams =  numpy.array([dfdH,  
						 dfdTm, 
						 dfdCp])

		else:
			jparams = numpy.array([dfdH, 
					       dfdTm])

		return jparams 

		'''if self.baseline == "floating":

			dfdm_f += (T/(numpy.exp((dH*(T/Tm - 1) + 
			          dCp*(Tm - T + T*numpy.log(T/Tm)))/(R*T)) + 1))	
		
			dfdy_int_f += (1/(numpy.exp((dH*(T/Tm - 1) + 
				      dCp*(Tm - T + T*numpy.log(T/Tm)))/(R*T)) + 1))

			dfdm_uf += (-T*(1/(numpy.exp((dH*(T/Tm - 1) + 
				   dCp*(Tm - T + T*numpy.log(T/Tm)))/(R*T)) + 1) - 1))

			dfdy_int_uf += (1 - 1/(numpy.exp((dH*(T/Tm - 1) + 
				       dCp*(Tm - T + T*numpy.log(T/Tm)))/(R*T)) + 1))
			
		jacobian =  numpy.array([dfdH, 
					 dfdTm, 
					 dfdCp, 
					 dfdm_f, 
			   		 dfdy_int_f, 
					 dfdm_uf, 
					 dfdy_int_uf])
		

		return jacobian'''

	def fit_params(self, 
               	       n_iter=100,
	               eps=1e-4, 
	               step=0.001, 
                       minimization="lm",
	               dampener = 1,
	               lm_scale=10):	

		count = 0
		old_error = 0
		jacobian = self.jparams() #jacobian is now 7xN	
		new_error = self.error_fun()
		wt_vector, wt_mat = generate_weights(self)
		old_error = 0

		while ((count < n_iter) and ( (abs(new_error-old_error) > eps))):
			

			if minimization == "gradient":

				shift = step*numpy.dot(wt_mat,numpy.dot(jacobian,d))

			elif minimization == "newton":
			
				#hessian will be a symmetric N_param x N_param matrix
				H = numpy.dot(jacobian,jacobian.T)
				inv_H = numpy.linalg.inv(H)
				shift = -step*numpy.dot(wt_mat,numpy.dot(inv_H,numpy.dot(jacobian,d)))

			elif minimization == "lm":

				H = numpy.dot(jacobian,jacobian.T)

				if self.dCp != 0:

					H = H + dampener*numpy.identity(3)*numpy.diagonal(H)

				else:
					H = H + dampener*numpy.identity(2)*numpy.diagonal(H)

				inv_H = numpy.linalg.inv(H)
				old_error = new_error
			
				shift = numpy.dot(inv_H,numpy.dot(jacobian,self.cd_signal-self.cd_calc()))
				self.update_params(shift)
				new_error = self.error_fun()

				if old_error > new_error:

					dampener /= lm_scale

				else:

					dampener *= lm_scale
			
				count += 1

		return

	def fit_baselines(self,
		  	  n_iter=200,
		  	  fit="points"):

		temp = self.temp
		Tm_check = self.Tm - 273.15
		n = len(temp)
		count = 0

		#preserve cell params
		dH = self.dH
		Tm = self.Tm
		nf = self.nf
		nu = self.nu

		for i in range(1,n):

			if temp[i] < Tm_check:
				count += 1

		nf_trial = count
		nu_trial = n - count
		error = self.error_fun()
		
		for i in range(2,nf_trial):

			for j in range(2,nu_trial):
			
				self.nf = i
				self.nu = j
				self.baselines()
				self.fit_params()
				err_check = self.error_fun()

				if err_check < error:
				
					nf = i
					nu = j
					self.baselines()
					Tm = self.Tm
					dH = self.dH
					error = err_check
				else:
					self.Tm = Tm
					self.dH = dH
				
		#restore cell with updated nf,nu if fit is good
		self.nf = nf
		self.nu = nu
		self.baselines()
		self.Tm = Tm
		self.dH = dH

		return 

	def error_fun(self):

		d = self.error*(self.cd_signal - self.cd_calc())
		error = numpy.dot(d,d)

		return error

	def update_params(self,shift):

		self.dH += shift[0]
		self.Tm += shift[1]

		if len(shift) > 2:
			self.dCp += shift[2]

		#cell.m_f += 0.1*shift[3]
		#cell.y_int_f += 0.1*shift[4]
		#cell.m_uf += 0.1*shift[5]
		#cell.y_int_uf += 0.1*shift[6]
		
		return

	def pickle_cell(self,filename):

		f = open(filename,'wb')
		pickle.dump(self,f)
		f.close()

		return

	def blank_cell(self,blank_cell_value):

		self.cd_signal = self.cd_signal - blank_cell_value

		return


	def params_to_file(self):

		with open(self.name, 'r') as file:
			data = file.readlines()
		
		data[1] = time.strftime("#modelled: %H:%M / %d %b %Y\n")
		data[2] = "#dH (kcal/mol): {0}\n".format(self.dH)
		data[3] = "#Tm (Kelvin): {0}\n".format(self.Tm)
		data[4]	= "#folded baseline pts: {0}\n".format(self.nf)
		data[5] = "#unfolded baseline pts: {0}\n".format(self.nu)

		with open(self.name, 'w') as file:
			file.writelines(data)

		return

class fit_session:

	def __init__(self, 
		     number_cells=5,
		     melt_start_number=1,
		     wavelength=223,
		     data_type="aviv"):

		self.number_cells = number_cells
		self.melt_start_number = melt_start_number
		self.wavelength = wavelength
		self.data_type = data_type
		
		
		return

	def generate_cells(self,blank=False):

		#creat a list of cell objects
		cell_list = []
	
		

		for n in range(0,self.number_cells):

			try:			
				cell_object = cell("cell{0}.csv".format(n),
					   	data_type=self.data_type)
			except IOError:
				print('No cell...generating blank entry instead')
				cell_object = []

			cell_list.append(cell_object)
			

		if blank:
			cell_list=[]*self.number_cells

		self.cell_list = cell_list


		return

	def fit_cells(self):

		for n in range(0,self.number_cells):

		    self.cell_list[n].fit_params()
		    self.cell_list[n].fit_baselines()
		    self.cell_list[n].params_to_file()

		return

	def combine_raw_data(self):

		total_signal = numpy.zeros([len(self.cell_list[0].temp),self.number_cells])
		
		for n in range(0,self.number_cells):

		    total_signal[:,n] = self.cell_list[n].cd_signal
			
		avg_cd_signal = numpy.mean(total_signal, axis=1)
		std_cd_signal = numpy.std(total_signal, axis=1)

		return avg_cd_signal, std_cd_signal

	def combine_frac_fold(self, filename='frac_fold.csv'):

		frac_fold = numpy.zeros([len(self.cell_list[0].temp),self.number_cells])
		temp_bin = numpy.zeros([len(self.cell_list[0].temp),self.number_cells])

		
		
		for n in range(0,self.number_cells):

		    frac_fold[:,n] = self.cell_list[n].frac_calc()
		    temp_bin[:,n] = self.cell_list[n].temp.T

		avg_frac_fold = numpy.mean(frac_fold, axis = 1)
		avg_temp = numpy.mean(temp_bin, axis = 1)
		
		f = open(filename, 'w')

		for n in range(0,len(avg_frac_fold)):

		    line_to_write = [str(avg_temp[n]).rjust(5),',', str(avg_frac_fold[n]).rjust(5), '\n']
	            f.writelines(line_to_write)

		f.close()		

		return avg_frac_fold

	def plot_cell(self,cell_number=0,label='insert_label'):

		current_cell = self.cell_list[cell_number]
		temp = current_cell.temp
		cd_signal = current_cell.cd_signal
		cd_calc = current_cell.cd_calc()
		
		ax = pylab.gca()

		pylab.plot(temp,cd_signal,'o',color='black')
                pylab.plot(temp,cd_calc,color='black')
		pylab.xlabel(r'Temperature ($^{\circ}$C)')
		pylab.ylabel('mdeg')
		pylab.ylim([-25,-4])
		dH = numpy.round(current_cell.dH, decimals=1)
		Tm = numpy.round(current_cell.Tm-273.15, decimals=1)
		nf = current_cell.nf
		nu = current_cell.nu
		textstr_dH = '${\Delta}H_{m}$ = %.1f kcal/mol' %dH
		textstr_Tm ='$T_{m}$ = %.1f $^{\circ}$C' %Tm
		textstr_nf ='$N_{folded}$ = %d' %nf
		textstr_nu ='$N_{unfolded}$ = %d'%nu
		ax.text(8,-6,textstr_dH, fontsize=16,ha='left',va='top')
		ax.text(8,-7.5,textstr_Tm, fontsize=16,ha='left',va='top')
		ax.text(8,-9,textstr_nf, fontsize=16,ha='left',va='top')
		ax.text(8,-10.5,textstr_nu, fontsize=16,ha='left',va='top')
		pylab.title(label)		
		pylab.show()

		return

	def cd_csv_creator(self): 
		   	 
		number_cells = self.number_cells
		melt_start_number = self.melt_start_number
		wavelength = self.wavelength

		def thermal_cd_output(filename,
			     	      WL):
	
			#reads through aviv dat files from
			#wavelength schedule experiment
	
			f = open(filename)		 
		
			for line in f:
	
				f = open(filename)
	
				if re.match(WL,line):

					cd_data = line;
			f.close()
	
			return cd_data

		wavelength = str(wavelength)

		# loop through cells

		for c in range(0,int(number_cells)):
		 
			d = melt_start_number
			output = open("cell{0}.csv".format(c),"w")
			output.writelines(time.strftime("#written: %H:%M / %d %b %Y\n"))
			output.writelines("#modeled: \n")
			output.writelines("#dH (kcal/mol): \n")
			output.writelines("#Tm (Kelvin): \n")
			output.writelines("#folded baseline pts: \n")
			output.writelines("#unfolded baseline pts: \n")
			output.writelines("#Wavelength(nm),CD_Signal(mdeg),Error(mdeg),Dynode(V),Temp(C)\n")
	
			for file in glob("*"):

				for file in glob("*"):
				
					if '#{0} Cell{1}.dat'.format(d,c) in file:
					 
						cd_data_line = thermal_cd_output(file,wavelength)
						cd_data_line = cd_data_line.split()
						#cd_data_line = map(float, cd_data_line)			
						#cd_data_line.append("\n")
						output.write(",".join(cd_data_line))
						output.write("\n")
						d += 1

			output.close()
			

		return 


def generate_weights(cell):

	dH = cell.dH
	Tm = cell.Tm
	#dCp = cell.dCp
	#m_f = cell.m_f 
	#y_int_f = cell.y_int_f 
	#m_uf = cell.m_uf 
	#y_int_uf = cell.y_int_uf 
	wt_vector = numpy.array([dH, Tm])#, dCp, m_f, y_int_f, m_uf, y_int_uf])
	wt_mat = numpy.identity(2)*wt_vector

	return wt_vector, wt_mat

def unpickle_cell(filename):

	f = open(filename,'rb')
	cell_object = pickle.load(f)
	f.close()
	
	return cell_object

