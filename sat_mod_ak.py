#!/usr/bin/env python

"""sat_mod_ak.py -- calculates model and satellite columns, including Averaging Kernal applications """

import numpy as np
import scipy.io
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import date as date
import matplotlib.pyplot as plt
import netCDF4 as nc4
#from bpch import bpch

import PseudoNetCDF as pnc
import sys

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, interp1d

from options_sat_mod_ak import get_options, check_options_valid, opts,replaceYYYYMMDD


class val_record:
    """A class for accumulating data of a parameter over a period"""
    
    def __init__(self,name,unit=""):
        self.name = name
        self.unit = unit
        self.data = []
        self.datelist = []
        
    def average(self):
        return(np.nanmean(self.data,axis=0))
    def std(self):
        return(np.nanstd(self.data,axis=0))
    def firstdate(self):
        return(min(self.datelist)) 

def pid(text):
    """Print text if global debug flag set to true"""
    if global_debug:
        print(text)

def frange(x, y, jump):
  """A simple utility for creating a sequentiual list of floats"""
  while x+jump <= y:
    yield x
    x += jump
    
def last_day_of_month(any_day):
    """Returns the last day of the month of a supplied date""" 
    next_month = any_day.replace(day=28) + td(days=4)  # this will never fail
    return next_month - td(days=next_month.day)
    
def fillAK(AK,i,j,AK_shape=(23,19)):
    """Produces a rectangular AK matrix given data"""
    #RAL's AKs are provided in a strange format. For a given coordinate ([i][j]), there
    #is a 1D list of components such that the nth component is AK[n][i][j]. These then
    #need to be converted into a rectangular matrix 
 
    this_AK_flat = np.zeros(AK_shape[0]*AK_shape[1]) #prepare empty flat array
    
    #read in AK values for given i,j point and fill this_AK_flat
    for n in range(0,AK_shape[0]*AK_shape[1]):
        this_AK_flat[n] = AK[n][i][j]
        pid(this_AK_flat[n])
    
    #convert this to rectangular array    
    this_AK = np.reshape(this_AK_flat,AK_shape,order='C')
    pid(this_AK)
    
    return this_AK #return rectangular array

def cut_to_domain(array,all_lats,all_lons,domain):
    """Cuts an array to only those values within a domain"""
    #2nd-last dimension of array is latitude
    #last dimension of array is longitude
    #domain = [south,north,west,east]
    
    
    #find lats/lons within domain. make numpy array
    good_lats = np.array([all_lats[k] <= domain[1] and all_lats[k] >= domain[0] for k in range(len(all_lats))])
    good_lons = np.array([all_lons[k] <= domain[3] and all_lons[k] >= domain[2] for k in range(len(all_lons))])
    
    #get array with out-of-bounds points flagged as -99999            
    flagged_lats = np.where(good_lats,all_lats,-99999.)
    flagged_lons = np.where(good_lons,all_lons,-99999.)
    
    #get indexes of edges of region        
    (min_lat_ix,max_lat_ix,
     min_lon_ix,max_lon_ix) = (min(np.ix_(flagged_lats != -99999.,flagged_lons != -99999.)[0].flatten()),
                               max(np.ix_(flagged_lats != -99999.,flagged_lons != -99999.)[0].flatten()),
                               min(np.ix_(flagged_lats != -99999.,flagged_lons != -99999.)[1].flatten()),
                               max(np.ix_(flagged_lats != -99999.,flagged_lons != -99999.)[1].flatten()))
    
    cut_array = array[...,min_lat_ix:max_lat_ix+1,min_lon_ix:max_lon_ix+1]
    
    return(cut_array)

def lateral_regrid(in_lat,in_lon,in_array,out_lat,out_lon):
    """Laterally regrids an array, interpolating where needed"""
    #in_lat,in_lon : latitudes and longitudes of the points on the input array
    #array : 2D array to regrid, 1st dimension is lat, 2nd is lon
    #out_lat,out_lon : latitudes and longitudes of the points on the output array
    
    #check dimensions are OK on input array, don't want extrapolations
    
    if max(out_lat) > max(in_lat) or min(out_lat) < min(in_lat) or max(out_lat) > max(in_lat) or min(out_lon) < min(in_lon):
        print("Error in lateral_regrid : output coordinates outside range of input coordinates")
        raise IOError
    
    #create interpolation function
    #interp_func = RegularGridInterpolator((in_lat, in_lon), in_array, method='linear')
    interp_func = RectBivariateSpline(in_lat,in_lon,in_array)
    
    out_array = interp_func(out_lat,out_lon)
    
    #create empty array
    #out_array = np.zeros((len(out_lat),len(out_lon)))
    
    #fill each point in empty array
    
    #out_array = interp_func(np.meshgrid(out_lat,out_lon))
    #for i in range(len(out_lat)):
    #    for j in range(len(out_lon)):
    #        out_array[i][j] = interp_func((out_lat[i],out_lon[j]))
    
    return out_array

def lateral_regrid_ND(in_lat,in_lon,in_array,out_lat,out_lon):
    """laterally regrids an N-dimensional array in its final two dimensions"""
    #in_lat,in_lon : latitudes and longitudes of the points on the input array
    #array : ND array to regrid, 1st is [N-1]th dimension is lat, Nth is lon
    #out_lat,out_lon : latitudes and longitudes of the points on the output array
    
    #create zeros array
    in_shape = list(in_array.shape)
    #replace the the last two elements here with length of out_lat and out_lon
    in_shape[len(in_shape)-2] = len(out_lat)
    in_shape[len(in_shape)-1] = len(out_lon)
    out_array = np.zeros(tuple(in_shape))
    
    #check shape of array
    dim_count = len(in_array.shape)
    
    if dim_count <= 2:
        print("Too few dimensions on in_array to lateral_regrid_ND")
        raise IOError
    
    #loop over 1st array
    for i in range(len(in_array)):        
        if dim_count == 3: #if 3D array:
            out_array[i] = lateral_regrid(in_lat,in_lon,in_array[i],out_lat,out_lon)
        else: #if higher dimensions, recall this function on each element 
            out_array[i] = lateral_regrid_ND(in_lat,in_lon,in_array[i],out_lat,out_lon)
    
    #return the filled output        
    return out_array

#def regrid_from_spacing(in_list,new_spacing):
#    """Produces a new set of points (i.e. longitudes or latitudes) covering the span of the input with the new spacing"""
#    return list(frange(min(in_list), max(in_list), spacing))

def conv4D_3D(in_array):
    """Checks if an array is 4D. If so, average across 1st dimension, if not, return unmodified"""
    num_dims = len(in_array.shape)
    if num_dims == 4:
        out_array = np.mean(GC_airdens,axis=0)
    elif num_dims == 3:
        out_array = in_array
    else:
        raise IOError("in conv4D_3D, in_array has %i dimensions. 4 or 3 required"%num_dims)
    
    return out_array
    

def get_amount_profile(GC_airdens,GC_mixrat,GC_height):
    """Produces a 3D array of the amount of a substance per area in each voxel"""
    
    air_amount = np.multiply(conv4D_3D(GC_airdens),
                             conv4D_3D(GC_height )
                            )
    tra_amount = np.multiply(air_amount,
                             conv4D_3D(GC_mixrat )
                            )
    return tra_amount
    
def get_cumulative(in_array):
    """Gets the cumulative sum of an array along its first dimension"""
    
    out_array = np.zeros_like(in_array)
    for i in range(len(in_array)):
        if i == 0: #if first
            out_array[i] = in_array[i]
        else:
            out_array[i] = np.add(out_array[i-1],in_array[i])
    return out_array
    
def interpolate_GC_to_MACC(GC_pressure,fixed_p_profile,GC_tra_amount):
    """vertically interpolates from GEOS-Chem's dynamic pressures to fixed profile"""
    
    #get cumulative amount of tracer. This makes the interpolation mass-conservative
    cum_tra_amount = get_cumulative(GC_tra_amount)
    
    #calculate mid point of pressure arrays, add top of atmosphere (minimum pressure) to end.
    #fixed
    fixed_p_mids = [0.5 * (fixed_p_profile[i]+fixed_p_profile[i+1]) for i in range(len(fixed_p_profile)-1)] + [min(fixed_p_profile)]

def is_record_day(this_date,cycle_type,end_date):
    """returns True/False if this is the last date of the cycle"""
    if this_date == end_date:
        return(True)
    if cycle_type == "day":
        return(True)
    if cycle_type == "month" and (this_date + td(days=1)).day == 1:
        return(True)
    if cycle_type == "season" and (this_date.strftime("%m%d") in ["0331","0630","0930","1231"]):
        return(True)
    return(False)
    
def is_read_day(this_date,option):
    """Returns True if today is a day to read in satellite data - either we're reading in daily data or
    we're doing monthly and it's the first day of the month"""
    if option.sat_time_res == "d":
        return(True)
    elif option.sat_time_res == "m" and this_date.day == 1:
        return(True)
    else:
        return(False)

def make_dict(option):
    """Creates an dictionary of empty val_record items for holding all the results"""

    this_dict = {}

    for spc in option.geos_species: #geos_chem species
        this_dict[spc] = val_record("GC_"+spc,unit="molec cm-2") #column
        this_dict[spc+"_GL"] = val_record("GC_"+spc+"_GL",unit="v/v") #ground level
    if option.do_geos_nox:
        this_dict["NOx"] = val_record("GC_NOx",unit="molec cm-2") #column
        this_dict["NOx_GL"] = val_record("GC_NOx_GL",unit="v/v")  #ground level
        
    if option.do_prodloss and option.geos_species != []:
        this_dict["prodOx"] = val_record("prod_Ox",unit="molec/cm3/s") #production of Ox
        this_dict["lossOx"] = val_record("loss_Ox",unit="molec/cm3/s") #loss of Ox       

    if option.do_gsc: #OMI ozone
        this_dict["GSC"]      = val_record("OMI O3",              "molec cm-2")
        if option.do_3D == False:
            this_dict["GSC_wBC"]      = val_record("OMI O3 with bias correction","molec cm-2")
            this_dict["BC"]      =      val_record("bias correction","molec cm-2")
            this_dict["GSC_wBC_old"]      = val_record("OMI O3 with OLD bias correction","molec cm-2")
            this_dict["BC_old"]      =      val_record("OLD bias correction","molec cm-2")
    
    if option.do_MACC: #MACC ozone
        this_dict["MACC"]     = val_record("MACC O3",             "molec cm-2")
    if option.do_MACC_wAK: #MACC ozone with averaging kernals
        this_dict["MACC_wAK"] = val_record("MACC O3 with OMI AKs","molec cm-2")
    if option.do_prior or option.do_geos_o3_wAK: #prior
        this_dict["prior"]    = val_record("prior",               "molec cm-2")

    if option.do_geos_o3_wAK: #GEOS-Chem ozone with averaging kernals
        this_dict["GC_O3_wAK"] = val_record("GC_O3 with OMI AKs",  "molec cm-2")
        
    if option.do_geos_o3_wAK:
        this_dict["AK"] = val_record("AKs",  "")
    
    return(this_dict)

def read_sat_to_dict(accum_dict,var,sat_data_dict,this_date,all_lat,all_lon,domain,do_3D=False):
    """Reads a variable from the satellite file and writes it, cut to domain, to accum_dict"""
        
    temp_dict = accum_dict
    temp_dict[var].datelist.append(this_date)
    
    
    if var == "AK": #the structure of AKs is a bit more complicated
        temp_dict[var].data.append(
                        get_AKs_from_sat(
                         sat_data_dict[var],
                         all_lat,all_lon,
                         domain
                         )
                        )
    elif var == "prior":
        temp_dict[var].data.append(
                               cut_to_domain(
                                get_av_from_sat(sat_data_dict[var],isprior=True,do_3D=do_3D),
                                all_lat,all_lon,
                                domain
                                )
                               )               
    
    else:
        temp_dict[var].data.append(
                               cut_to_domain(
                                get_av_from_sat(sat_data_dict[var],do_3D=do_3D),
                                all_lat,all_lon,
                                domain
                                )
                               )   
    
    return(temp_dict) 
            

def get_av_from_sat(RAL_input,DU_conv=True,isprior=False,do_3D=False):
    """Return the average gridded values from RAL-format data"""
    

    
    if DU_conv:
        #if True, convert from DU to molec cm-2
        conv = 2.687e16
    else:
        conv = 1.
    
    if do_3D: #3D, read each layer individually
                
        if isprior: #For non-prior fields, level 0 is full column and shoul be skipped
            first_idx = 0
        else:
            first_idx = 1        
        
        av_col = np.zeros((len(RAL_input[0])-first_idx,len(RAL_input[0][0].T1),len(RAL_input[0][0].T1[0])))
        
        for k in range(first_idx,len(RAL_input[0])): #level 0 is whole column, skip it
            av_col[k-1] = np.divide(np.array(RAL_input[0][k].T1),np.array(RAL_input[0][k].N))*conv
    else: #2D  
        if isprior: #prior uses index 0, not 1
            lev_idx = 0
        else:
            lev_idx = 1
          
        av_col = np.divide(np.array(RAL_input[0][lev_idx].T1),np.array(RAL_input[0][lev_idx].N))*conv
    
    av_col[av_col == 0.] = np.nan #mark non-detections as np.nan
    
    return av_col

def get_AKs_from_sat(RAL_input,all_lat,all_lon,domain):
    """Returns the AKs as a flat array for each lat lon point from RAL-format data"""
        
    AKs_T1 = np.array(RAL_input[0][:].T1)
    AKs_N  = np.array(RAL_input[0][:].N)
    
    #these will have 3 dimesions:
    #0: the 437 numbers of the AK
    #1: latitude
    #2: longitude 
    
    #dimension sizes
    num_AKs = len(AKs_T1) #should be 19*23=437
    num_lats= len(all_lat)
    num_lons= len(all_lon)
    
    #print AKs_T1[0].shape
    
    #check that these lat/lon lengths equal the dimensions sizes of the input
    if num_lats != AKs_T1[0].shape[0] or num_lons != AKs_T1[0].shape[1]:
        print("Error in get_AKs_from_sat. Dimension sizes unequal")
        print("len(all_lat) = %i" %len(all_lat))
        print("len(all_lon) = %i" %len(all_lon))
        print("AKs_T1.shape = (next line)")
        print(AKs_T1.shape)
        print(AKs_T1[0].shape)
        raise IOError("Invalid shape")
    
    #create a simple numpy array                    
    AKs_arrayi = np.zeros((num_AKs,num_lats,num_lons))
    for a in range(num_AKs):
        for b in range(num_lats):
            for c in range(num_lons):
                AKs_arrayi[a][b][c] = np.divide(AKs_T1[a][b][c],AKs_N[a][b][c])
    
    #remove non-detections
    AKs_arrayi[AKs_arrayi == 0.] = np.nan
    
    #cut to domain
    AKs_array_cut = cut_to_domain(AKs_arrayi,all_lat,all_lon,domain)
    
    #re-arrage so that AK list dimension is last.
    AKs_array_cut_rearr = np.zeros((AKs_array_cut.shape[1],AKs_array_cut.shape[2],AKs_array_cut.shape[0]))
    for a in range(AKs_array_cut.shape[0]): #AK list dim
        for b in range(AKs_array_cut.shape[1]): #lat
            for c in range(AKs_array_cut.shape[2]): #lon
                AKs_array_cut_rearr[b][c][a] = AKs_array_cut[a][b][c]
    
    return AKs_array_cut_rearr

def pressure_cutoff(amount,pressure,cutoff=450.,commentary="",do_avg=False):
    """Determines the amount of substance below the pressure cutoff, using interpolation where needed"""
    #amount : 3D grid of amount of substance (vertical, lat, lon)
    #pressure : 3D grid of pressure at the *bottom* of each box (vertical, lat, lon)
    #cutoff : is the pressure cutoff, same units as previous variable.
    #commentary : set to any string to get a printout when this command is running (useful for diagnostic)
    
    #diagnostic
    if commentary != "":
        print("%s - Calculating pressure cutoff"%commentary)
    
    #work out shape of array
    (num_levels,num_lats,num_lons) = amount.shape
          
    #prepare output grid
    output = np.zeros((num_lats,num_lons))
    
    
    for i in range(num_lats):
        for j in range(num_lons):
            pres_col = pressure[:,i,j] #get 1D array of pressures at this i,j point
            amount_col = amount[:,i,j] #get 1D array of amounts at this i,j point
            amount_col = list(np.where(np.isnan(np.array(amount_col)),0.,np.array(amount_col)))
            #for each i,j point, get the sum up to this level (linearlt interpolating using log pressures)
            output[i,j] = float(log_interp_sum_to_level(pres_col,amount_col,cutoff))
            if commentary == "full" and i == 20 and j == 20:
                print pres_col
                print amount_col
                print output[i,j]
    
    return(output)

def simple_fixed_regrid_19_10(inp):
    """Bodge function to regrid from 19 layers to 10"""
    
    in_shp = np.array(inp).shape
    
    output = np.zeros((in_shp[0],9,in_shp[2],in_shp[3]))
    
    for t in range(in_shp[0]):
        
        output[t][0] = inp[t][0]
        output[t][1] = inp[t][1]
        output[t][2] = np.add(inp[t][2],inp[t][3])
        output[t][3] = np.add(inp[t][4],inp[t][5])
        output[t][4] = np.add(inp[t][6],inp[t][7])
        output[t][5] = np.add(inp[t][8],inp[t][9])
        output[t][6] = np.add(inp[t][10],inp[t][11])
        output[t][7] = np.add(inp[t][12],inp[t][13])
        output[t][8] = np.add(inp[t][14],np.add(inp[t][15],np.add(inp[t][16],np.add(inp[t][17],inp[t][18]))))
    
    return(output)

    
def pressure_slicer(amount,pressure,levels=[
         1.00000000e+03,    
                           4.50000000e+02,   
         1.70000015e+02,   1.00000000e+02,   4.99999962e+01,
         2.99999981e+01,   2.00000019e+01,   1.00000000e+01,
         5.00000095e+00,   3.00000072e+00,   1.99999964e+00,
         1.00000000e+00,   5.00000060e-01,   3.00000072e-01,
         1.70000017e-01,   1.00000001e-01,   4.99999821e-02,
         3.00000068e-02,   1.69999916e-02,   9.99999978e-03]
                                           ,commentary=""):
                                           
    """Determines the amount of substance within different layers of the atmosphere, using interpolation where needed"""
    #amount : 3D grid of amount of substance (vertical, lat, lon)
    #pressure : 3D grid of pressure at the *bottom* of each box (vertical, lat, lon)
    #levels : fixed pressure levels to report results on (output[0] will be between levels[0] and [1] etc.) 
    #commentary : set to any string to get a printout when this command is running (useful for diagnostic)
    
    #diagnostic
    if commentary != "":
        print("%s - Calculating on pressure levels"%commentary)
    
    #work out shape of array
    (num_levels,num_lats,num_lons) = amount.shape
          
    #prepare output grid
    cumulative = np.zeros((num_lats,num_lons,len(levels)))
    output = np.zeros((len(levels),num_lats,num_lons))
    
    
    for i in range(num_lats):
        for j in range(num_lons):
            pres_col = pressure[:,i,j] #get 1D array of pressures at this i,j point
            amount_col = amount[:,i,j] #get 1D array of amounts at this i,j point
            #for each i,j point, get the sum up to this level (linearlt interpolating using log pressures)
            cumulative[i][j] = log_interp_sum_to_level(pres_col,amount_col,levels)
            for k in range(len(levels)):
                if k == 0:
                    output[k][i][j] = cumulative[i][j][k]
                else:
                    output[k][i][j] = cumulative[i][j][k] - cumulative[i][j][k-1]
    
    return(output)    

def log_average(a,b):
    """returns logatithmic average of two numbers"""
    c = np.exp(1)**((np.log(a)+np.log(b))*0.5)
    return(c)
    
def log_interp_sum_to_level(pres_col,amount_col,pres_points,toa_pres=0.01):
    """linearly interpolates cumulative of amount_col at pres_points based on the logarithm of pres_col"""
    
    #make sure pres points is an iterable object (list or np array)
    #also get logarithms
    if type(pres_points) == float: #if we've passed in a single number
        pres_points_log = [np.log(pres_points)]
    else:
        pres_points_log = np.log(pres_points)
    
    #pres_col is box bottom pressures, so append in top of atmosphere pressure. 
    pres_col = np.append(pres_col,toa_pres)
    #logarithm   
    pres_col_log = np.log(pres_col)
    
    #get cumulative version of amount
    amount_col_cum = get_cumulative(amount_col)
    amount_col_cum = np.append(0,amount_col_cum)
    
    #move any points below lowest and above higest to edge of range 
    if type(pres_points_log) != float:
        for i in range(len(pres_points_log)):
            if pres_points_log[i] < min(pres_col_log):
                pres_points_log[i] = min(pres_col_log)
            elif pres_points_log[i] > max(pres_col_log):
                pres_points_log[i] = max(pres_col_log)
    
    #interpolate
    f = interp1d(pres_col_log,amount_col_cum)
    
    try:
        out_amounts = f(pres_points_log)
    except ValueError:
        print("ERROR in log_interp_sum_to_level. Printing debug info")
        print("--Pres col log--")
        print(pres_col_log)
        print("----------------")
        print("--pres_points_log--")
        print(pres_points_log)
        print("-------------------")
        sys.exit()
    
    return(out_amounts)
    
    
    
def apply_AKs_grid(o3,AKs,GC_pressures,toa=0.01,debug=False):
    """Appies AKs to a model o3 grid"""
    
    #get dimensions of model grid
    (mod_extent_k,mod_extent_i,mod_extent_j) = o3.shape
    
    #define the fixed pressure profiles, in hPa
    #high-res in troposphere
    HR_p_profile = [
         1.00000000e+03,   8.60000000e+02,   7.50000000e+02,
         6.00000000e+02,   4.50000000e+02,   3.10000000e+02,
         1.70000015e+02,   1.00000000e+02,   4.99999962e+01,
         2.99999981e+01,   2.00000019e+01,   1.00000000e+01,
         5.00000095e+00,   3.00000072e+00,   1.99999964e+00,
         1.00000000e+00,   5.00000060e-01,   3.00000072e-01,
         1.70000017e-01,   1.00000001e-01,   4.99999821e-02,
         3.00000068e-02,   1.69999916e-02,   9.99999978e-03]
    HR_levs = len(HR_p_profile) - 1 #number of levels
    #low-res in troposphere
    LR_p_profile = [
         1.00000000e+03,    
                           4.50000000e+02,   
         1.70000015e+02,   1.00000000e+02,   4.99999962e+01,
         2.99999981e+01,   2.00000019e+01,   1.00000000e+01,
         5.00000095e+00,   3.00000072e+00,   1.99999964e+00,
         1.00000000e+00,   5.00000060e-01,   3.00000072e-01,
         1.70000017e-01,   1.00000001e-01,   4.99999821e-02,
         3.00000068e-02,   1.69999916e-02,   9.99999978e-03]
    LR_levs = len(LR_p_profile)   
    
    #generate output grid
    output = np.zeros((LR_levs-1,mod_extent_i,mod_extent_j))

    
    #cycle through points
    for i in range(mod_extent_i):
        for j in range(mod_extent_j):
            this_col_geos_o3 = np.array([]) #empty 1D array for column
            this_col_geos_p  = np.array([]) #empty 1D array for pressures
            
            #print mod_extent_k
            
            if debug and i == 10 and j == 10:
                debug_here = True 
            else:
                debug_here = False
            
            for k in range(mod_extent_k):
                this_col_geos_o3 = np.append(this_col_geos_o3,o3[k][i][j]) #fill with amount of o3
                #for pressure, interpolate logarithmically between pressure_bot of this and next column
                if k != (mod_extent_k-1):
                    this_col_geos_p = np.append(this_col_geos_p,log_average(GC_pressures[k][i][j],GC_pressures[k+1][i][j]))
                else: #at top of atmopshere assume next level "bottom pressure" is toa
                    this_col_geos_p = np.append(this_col_geos_p,log_average(GC_pressures[k][i][j],toa))
                    
            #now, for this column, interpolate (using log pressures) onto HR pressure grid            
            sum_to_level = log_interp_sum_to_level(this_col_geos_p,this_col_geos_o3,HR_p_profile,toa_pres=0.01)
            #sum_to_level[0] should be zero (sum up to 1000hPa)
            #Otherwise, turn this into amount in each layer
            amount_in_level = np.zeros(len(sum_to_level)-1)
            
            for k in range(1,len(sum_to_level)):
                amount_in_level[k-1]=sum_to_level[k]-sum_to_level[k-1]
            
            #do the AKs on this column                       
            post_AK_col = apply_AKs_col(amount_in_level,AKs[i][j])    
            
            #Fill in this output
            for k in range(LR_levs-1):
                output[k][i][j] = post_AK_col[k]
            
            if debug_here:
                print("----GEOS O3 amounts, GEOS grid-----")
                print(this_col_geos_o3)
                print("-----------------------------------")
                print("----GEOS pressures,  GEOS grid-----")
                print(this_col_geos_p)
                print("-----------------------------------")
                print("----GEOS O3 amounts, HR grid-------")
                print(amount_in_level)
                print("-----------------------------------")
                print("----HR pressure levels ------------")
                print(HR_p_profile)
                print("-----------------------------------")
                print("---AKs, unspun---------------------")
                print(AKs[i][j])
                print("-----------------------------------")
                print("---Post AK col, on LR grid---------")
                print(post_AK_col)
                print("-----------------------------------")
                print("---LR pressure levels--------------")
                print(LR_p_profile)
                print("-----------------------------------")
    
    #set missing to nans
    for l in range(len(output)):
        for i in range(len(output[0])):
            for j in range(len(output[0][0])):
                #if l==0:
                #    print "Output l = %i, i = %i, j = %i is %g" %(l,i,j,output[l][i][j]) 
                if output[l][i][j] == 0.:                    
                    output[l][i][j] = np.nan
                   
    return output
                
                
def apply_AKs_col(amount_in_level,AKs):
    """Multiplies the input column by the AK"""
    AK_matrix = fillAK(AKs)
    amount_out_level = np.dot(amount_in_level,AK_matrix)
    return amount_out_level                

def fillAK(AK,debug=False):
    """Unravel a 437 list AK"""
    this_AK_flat = np.zeros(23*19)
    #clock = 0
    for n in range(0,23*19):
        this_AK_flat[n] = AK[n]
        if debug:
            print(this_AK_flat[n])
    this_AK = np.reshape(this_AK_flat,(23,19),order='C')
    if debug:
        print(this_AK)
    return this_AK
    
def bias_correct(array,lat,date,BCs="new"):
    
    latbands = [-75.,-45.,-15.,15.,45.,75.]
    
    
    
    if BCs == "new":    
        corrs= np.multiply(
               np.array([[-0.431,-0.911,-0.365,-3.41,  -3.51,  -2.22, -1.9,  -2.94, -4.62,  -2.82,  -3.01, -2.04 ],
                         [-5.88, -5.88, -6.6,  -3.94,  -2.58,  -2.02, -1.25, -0.336,-0.68,   1.84,   0.015,-2.9  ],
                         [-7.73, -7.53, -8.64, -7.74,  -6.68,  -5.03, -4.36, -4.15, -4.97,  -4.44,  -4.97, -6.48 ],
                         [-6.88, -8.48, -9.92, -9.04,  -7.07,  -4.73, -4.27, -4.19, -4.62,  -3.58,  -2.62, -3.49 ],
                         [-5.64, -4.86, -9.86, -6.65,  -4.51,  -2.01, -2.33, -0.923,-0.636,  0.0876,-1.09, -2.03 ],
                         [-4.24, -3.48, -5.82,  0.363,  0.0274, 0.0176,0.776, 2.69,  1.14,  -2.64,  -7.44, -4.01 ]
                      ]),2.687e16) #in molec.cm-2
    else:
            corrs= np.multiply(
               np.array([[-1.35, -1.44, -1.6 , -3.71, -3.37, -2.48, -2.26, -2.9 ,-2.57,-1.1 ,-1   ,-1.25],
                         [-5.72, -5.7 , -5.89, -4.14, -2.97, -2.29, -2.16, -1.05,-0.67,1.46 ,-1.15,-2.52],
                         [-5.99, -5.38, -5.63, -5.04, -4.15, -2.51, -2.34, -1.74,-2.65,-1.24,-2.34,-2.79],
                         [-5.34, -6.28, -6.85, -6.4 , -5.01, -3.01, -2.95, -2.24,-2.74,-0.97,-1.27,-1.64],
                         [-6.94, -7.11, -9.69, -8.18, -6.69, -4.28, -3.18, -1.13,-1.87,-0.70,-2.79,-3.97],
                         [-9.28, -8.04, -9.02, -5.44, -6.52, -4.88, -3.36, -1.65,-2.18,-3.75,-9.05,-5.77]
                      ]),2.687e16) #in molec.cm-2
                  
                  
    corr_m = date.month-1
    
    this_month_corrs = [corrs[i][corr_m] for i in range(len(latbands))]
    
    out_array = np.zeros_like(array)      
    for i in range(len(lat)):
         if lat[i] <= -75.:
            thiscorr = corrs[0][corr_m]
         elif lat[i] >= 75.:
            thiscorr = corrs[5][corr_m]
         else:
            interp = scipy.interpolate.interp1d(latbands,this_month_corrs)
            thiscorr = interp(lat[i])
         #print "Correction for lat = %g, month = %i  -> subtract %g"%(lat[i]+1,corr_m,thiscorr)
         #print array[i]
         out_array[i] = np.subtract(array[i],thiscorr)
         #print out_array[i]

    return out_array  


def nanlike(nparray):
    z = np.empty(nparray.shape)
    z[:] = np.nan
    return z

    
###MAIN FUNCTION STARTS HERE###
#==options==
option = opts(*get_options())

#check options
check_options_valid(option)

#are we using satellite data?
use_sat_data = option.do_gsc or option.do_MACC or option.do_MACC_wAK or option.do_prior or option.do_o3_wAK

#are we using model output?
if option.geos_species != []:
    use_mod_data = True
else:
    use_mod_data = False

#define output levels #this should be in options, but its here for now.
out_levs = [450.,170.,50.,20.,5.,2.,0.5,0.17,0.001]

#create dictionary for holding results.

result_dict = make_dict(option)
             
#Begin daily loop
this_date = option.start_date
days_since_average = 0


while this_date <= option.end_date:
    print("Processing %s" %this_date.strftime("%Y%m%d"))
    #if either this is the first date, or yesterday was a record date, create a new dictionary for results up to this point
    if this_date == option.start_date or is_record_day(this_date - td(days=1),option.cycle_type,option.end_date):
        accum_dict = make_dict(option)
        o3_full_profile = [] #also this small array
        pressure_bot_full_profile = [] #also this small array, won't be used unless AKs read monthly
        
    #read in satellite data.
    if use_sat_data and is_read_day(this_date,option):
        print("Reading satellite data")
        satf_path = replaceYYYYMMDD(option.sat_data_path,this_date)
        try:
            satf = scipy.io.readsav(satf_path,python_dict=True)
        except IOError:
            print("This date has no file of satellite data. Skipping")         
            this_date += td(days=1)
            continue
        
        #it's possible to have empty files (no valid passes)               
        try:
            null = satf['x'].LAG[0][0] #will raise exception if no data for day
        except AttributeError: #invalid file
            print("This date has a file for satellite data, but it is empty or corrupt. Skipping")         
            this_date += td(days=1)
            continue
        
        #get full latitudes and longitudes from file      
        all_lat = np.array(list(frange(satf['x'].LAG[0][0]+0.5*satf['x'].LAG[0][2],satf['x'].LAG[0][1],satf['x'].LAG[0][2]))) #failure will occur here if invalid file        
        all_lon = np.array(list(frange(satf['x'].LOG[0][0]+0.5*satf['x'].LOG[0][2],satf['x'].LOG[0][1],satf['x'].LOG[0][2])))
        #dictionary of satellite data
        sat_data_dict = {
            "GSC":satf['x'].GSC,
            "MACC":satf['x'].MSC,
            "MACC_wAK":satf['x'].MSC_AK_HR,
            "prior":satf['x'].IMAK_APR_SC,
            "AK":satf['x'].AK_RSC_TSC
            }
        
        
        if option.do_gsc: #OMI ozone
            accum_dict = read_sat_to_dict(accum_dict,"GSC",sat_data_dict,this_date,all_lat,all_lon,option.domain,do_3D=option.do_3D)      
        if option.do_MACC: #MACC ozone
            accum_dict = read_sat_to_dict(accum_dict,"MACC",sat_data_dict,this_date,all_lat,all_lon,option.domain,do_3D=option.do_3D)
        if option.do_MACC_wAK: #MACC ozone with averaging kernals
            accum_dict = read_sat_to_dict(accum_dict,"MACC_wAK",sat_data_dict,this_date,all_lat,all_lon,option.domain,do_3D=option.do_3D)
        if option.do_prior or option.do_geos_o3_wAK: #prior
            accum_dict = read_sat_to_dict(accum_dict,"prior",sat_data_dict,this_date,all_lat,all_lon,option.domain,do_3D=option.do_3D)
        if option.do_geos_o3_wAK: #averaging kernals
            accum_dict = read_sat_to_dict(accum_dict,"AK",sat_data_dict,this_date,all_lat,all_lon,option.domain)
        
        if not option.do_3D:
            accum_dict_num = len(accum_dict["GSC"].data)
            good_lats = all_lat[np.array([all_lat[k] <= option.domain[1] and all_lat[k] >= option.domain[0] for k in range(len(all_lat))])]
            print good_lats
            
            accum_dict["GSC_wBC"].datelist.append(accum_dict["GSC"].datelist[accum_dict_num-1]) #date
            accum_dict["GSC_wBC_old"].datelist.append(accum_dict["GSC"].datelist[accum_dict_num-1]) #date                   
            accum_dict["BC"].datelist.append(accum_dict["GSC"].datelist[accum_dict_num-1])      #date
            accum_dict["BC_old"].datelist.append(accum_dict["GSC"].datelist[accum_dict_num-1])      #date
            
            accum_dict["GSC_wBC"].data.append(bias_correct(accum_dict["GSC"].data[accum_dict_num-1],good_lats,accum_dict["GSC_wBC"].datelist[accum_dict_num-1])) #bias correction for this data
            accum_dict["GSC_wBC_old"].data.append(bias_correct(accum_dict["GSC"].data[accum_dict_num-1],good_lats,accum_dict["GSC_wBC_old"].datelist[accum_dict_num-1],BCs="old")) #bias correction for this data USING OLD VALUES
            accum_dict["BC"].data.append(np.subtract(accum_dict["GSC_wBC"].data[accum_dict_num-1],accum_dict["GSC"].data[accum_dict_num-1]))
            accum_dict["BC_old"].data.append(np.subtract(accum_dict["GSC_wBC_old"].data[accum_dict_num-1],accum_dict["GSC"].data[accum_dict_num-1]))
                
        yesterday_sat_data_dict = sat_data_dict
        #print accum_dict["GSC"].data
    
    #read in model data
    if use_mod_data: #this will be read every day, regardless of read/write options
        print("Reading model data")
        modf_path = replaceYYYYMMDD(option.mod_data_path,this_date) #file
        #modf = bpch(modf_path)
        
        modf = pnc.pncopen(modf_path, format = 'geoschemfiles.bpch')
        #get some values used for all
        #lat and lon
        mod_lat = np.array(modf.variables['latitude'])
        mod_lon = np.array(modf.variables['longitude'])
        
        if use_mod_data and use_sat_data:
            #get satellite latitudes and logitudes in domain
            new_lat = [all_lat[i] for i in range(len(all_lat)) if all_lat[i] >= option.domain[0] and all_lat[i] <= option.domain[1]]
            new_lon = [all_lon[i] for i in range(len(all_lon)) if all_lon[i] >= option.domain[2] and all_lon[i] <= option.domain[3]]

        
        #air density and amounts
        box_height = np.array(modf.variables['BXHGHT-$_BXHEIGHT'])[0] * 100 #box height in cm      
        air_den =    np.array(modf.variables['TIME-SER_AIRDEN'])[0]  # air density molec.cm-3  
        air_amount = np.multiply(box_height,air_den) #air per grid box molec.cm-2
                
        #data for pressure
        pressure_bot = np.array(modf.variables['PEDGE-$_PSURF'])[0]  # pressure  
        
        if use_mod_data and use_sat_data: #laterally regrid if needed
            pressure_bot = lateral_regrid_ND(mod_lat,mod_lon,pressure_bot,new_lat,new_lon)
        
        if option.do_prodloss:
            #read in prodloss
            print("Reading prodloss data")
            prodloss_path = replaceYYYYMMDD(option.prodloss_path,this_date) #file
            #prodlossf = bpch(prodloss_path)          
            prodlossf = pnc.pncopen(prodloss_path, format = 'geoschemfiles.bpch')
                      
        for spc in option.geos_species:
            
            #For each model species, read in data
            spc_mixratio = np.array(modf.variables["IJ-AVG-$_%s"%spc])[0] * 1e-9 #array of mixing ratios 
            
                      
            #column
            spc_molec_per_cm2 = np.multiply(air_amount,spc_mixratio) #molec per cm2 for each box
            
            if spc == "O3":
                pass
                #print("DEBUG")
                #print spc_mixratio.shape
                #print spc_molec_per_cm2.shape
            
            #If we're doing both satellite and model data, we'll need to laterally regrid the model data
            if use_mod_data and use_sat_data:                
                spc_molec_per_cm2 = lateral_regrid_ND(mod_lat,mod_lon,spc_molec_per_cm2,new_lat,new_lon)
                spc_mixratio =      lateral_regrid_ND(mod_lat,mod_lon,spc_mixratio,new_lat,new_lon)
            
            if spc == "O3" and option.do_geos_o3_wAK:
                o3_full_profile.append(spc_molec_per_cm2)
                pressure_bot_full_profile.append(pressure_bot)
            
            accum_dict[spc].datelist.append(this_date) #add date for this variable
            if option.do_3D: #3D layers for species
                amount_in_level = pressure_slicer(spc_molec_per_cm2,pressure_bot,levels=out_levs)
                accum_dict[spc].data.append(amount_in_level) #add to dictionary
            else: #just surface-450hPa                             
                spc_tropospheric = pressure_cutoff(spc_molec_per_cm2,pressure_bot) #get amount in troposphere
                accum_dict[spc].data.append(spc_tropospheric) #add to dictionary
            
            #ground_level
            spc_gl_ppb   = spc_mixratio[0] #species mixing ratio at ground level
            accum_dict[spc+"_GL"].datelist.append(this_date) #add date for this variable
            accum_dict[spc+"_GL"].data.append(spc_gl_ppb) #add to dictionary
        
        if option.do_geos_nox: #if we're doing NOx
            GC_time_index = accum_dict["NO"].datelist.index(this_date)
            accum_dict["NOx"].datelist.append(this_date)
            accum_dict["NOx"].data.append(np.add(accum_dict["NO"].data[GC_time_index],
                                                 accum_dict["NO2"].data[GC_time_index]))
            accum_dict["NOx_GL"].datelist.append(this_date)
            accum_dict["NOx_GL"].data.append(np.add(accum_dict["NO_GL"].data[GC_time_index],
                                                 accum_dict["NO2_GL"].data[GC_time_index]))
        
        if option.do_prodloss:
            GC_time_index = accum_dict["O3"].datelist.index(this_date)
            
            
            oxprod = np.array(prodlossf.variables["PORL-L=$_POx"])[0] #O3 production, in kg/m3/s
            oxloss = np.array(prodlossf.variables["PORL-L=$_LOx"])[0] #O3 loss, in 1/m3/s
            
            
            #print oxloss[0][50]
            #print "DEBUG preA"
            #replace "inf" with np.nan         
            oxloss = np.where(np.isinf(oxloss),np.nan,oxloss)
            #print oxloss[0][50]
            #print "DEBUG A"            
            #oxprod and oxloss are on 38 layers.
            #Add nine zero layers at top
            
            pl_shape = oxprod.shape
            nine_0 = np.zeros((9,pl_shape[1],pl_shape[2]))
            
            oxprod = np.concatenate((oxprod,nine_0),axis=0)
            oxloss = np.concatenate((oxloss,nine_0),axis=0)
            
            print oxprod.shape
            
            #let's get things into sensible units
            #Prod - starts in kg/m3/s
            oxprod = oxprod * 1e3 #in g/m3/s
            oxprod = oxprod * 1e-6 #in g/cm3/s
            oxprod = oxprod * (6.022e23/58.) # in molec/cm3/s         
            oxprod_per_cm2 = np.multiply(box_height,oxprod) #molec per cm2 per sec for each box
            
            #Loss - starts in 1/m3/s (fraction of box's o3 lost per m3 per sec) 
            #print oxloss[0][50]
            #print "DEBUG B"
            #get areas of boxes
            area = 8.6e8 #m2 BODGE flat value.
            oxloss = oxloss * area #1/m/s
            #print oxloss[0][50]
            #print "DEBUG C"
            oxloss = np.multiply(oxloss,box_height) * 0.01 #1/s
            #print oxloss[0][50]
            #print "DEBUG C"
            oxloss_per_cm2 = np.multiply(oxloss,
                                 np.multiply(np.array(modf.variables["IJ-AVG-$_O3"])[0] * 1e-9,
                                             air_amount)
                                ) #molec per cm2 per sec for each box
            
            #print oxloss_per_cm2[0][50]
            #print "DEBUG D"

                       
            #lateral regrid
            if use_mod_data and use_sat_data:                
                oxprod_per_cm2 =      lateral_regrid_ND(mod_lat,mod_lon,oxprod_per_cm2,new_lat,new_lon)
                oxloss_per_cm2 =      lateral_regrid_ND(mod_lat,mod_lon,oxloss_per_cm2,new_lat,new_lon)
            
            
            accum_dict["prodOx"].datelist.append(this_date) #date for variable
            accum_dict["lossOx"].datelist.append(this_date) #date for variable
            if option.do_3D: #3D layers for species
                oxprod_in_level = pressure_slicer(oxprod_per_cm2,pressure_bot,levels=out_levs)
                oxloss_in_level = pressure_slicer(oxloss_per_cm2,pressure_bot,levels=out_levs)
                accum_dict["prodOx"].data.append(oxprod_in_level) #add to dictionary
                accum_dict["lossOx"].data.append(oxloss_in_level) #add to dictionary
            else: #just surface-450hPa
                
                #print oxloss[0][50]
                #print "DEBUG E"
                                             
                oxprod_tropospheric = pressure_cutoff(oxprod_per_cm2,pressure_bot) #get amount in troposphere
                oxloss_tropospheric = pressure_cutoff(oxloss_per_cm2,pressure_bot) #get amount in troposphere
                
                #print oxloss_tropospheric.shape
                #print oxloss_tropospheric
                #print "DEBUG F"
                
                accum_dict["prodOx"].data.append(oxprod_tropospheric) #add to dictionary
                accum_dict["lossOx"].data.append(oxloss_tropospheric) #add to dictionary           
    
    if option.do_geos_o3_wAK:
        #If we're using daily satellite data, we apply the daily AKs each day then take the average of the result at the end
        if option.sat_time_res == "d":
            print("Applying AKs")
            AK_time_index = accum_dict["AK"].datelist.index(this_date)
            GC_time_index = accum_dict["O3"].datelist.index(this_date)
            
            if option.do_3D:
                o3_wAK_noprior = apply_AKs_grid(o3_full_profile[GC_time_index],
                                                accum_dict["AK"].data[AK_time_index],
                                                pressure_bot_full_profile[GC_time_index])
            else:
                o3_wAK_noprior = apply_AKs_grid(o3_full_profile[GC_time_index],
                                                accum_dict["AK"].data[AK_time_index],
                                                pressure_bot_full_profile[GC_time_index])[0]  #only lowest layer             
            
            accum_dict["GC_O3_wAK"].datelist.append(this_date)
            accum_dict["GC_O3_wAK"].data.append(o3_wAK_noprior)
            
                                            
        elif option.sat_time_res == "m" and ((this_date + td(days=1)).day == 1 or this_date == option.end_date): #if we read monthly data AND it's the last day of month or the last date
            print("Applying AKs")
            
            o3_full_profile_month_average = np.average(o3_full_profile,axis=0)
            del o3_full_profile #reset this array, it's done its job
            o3_full_profile = [] 
                       
            pressure_bot_full_profile_month_average = np.average(pressure_bot_full_profile,axis=0)            
            del pressure_bot_full_profile #reset this array, it's done its job
            pressure_bot_full_profile = []
            
            start_month_date = this_date.replace(day=1)
            AK_time_index = accum_dict["AK"].datelist.index(start_month_date)
            
            if option.do_3D:
                o3_wAK_noprior = apply_AKs_grid(o3_full_profile_month_average,
                                                  accum_dict["AK"].data[AK_time_index],
                                                  pressure_bot_full_profile_month_average)
            else:
                o3_wAK_noprior = apply_AKs_grid(o3_full_profile_month_average,
                                                  accum_dict["AK"].data[AK_time_index],
                                                  pressure_bot_full_profile_month_average)[0]  #only lowest layer                            
            
            accum_dict["GC_O3_wAK"].datelist.append(this_date)
            accum_dict["GC_O3_wAK"].data.append(o3_wAK_noprior)
            
    #Now, to write the results to results_dict, averaging if need be.
    if is_record_day(this_date,option.cycle_type,option.end_date):
        print("Recording to result dictionary")
        #loop over everything in accum_dict, taking average over time
        for key in result_dict:
                     
            result_dict[key].datelist.append(min(accum_dict[key].datelist)) #set date to first date recorded for this variable
            
            #if option.mask_as_all:
            #    data_masked = np.ma.array(accum_dict[key].data, mask = np.logical_or(np.isnan(accum_dict[key].data),np.isnan(accum_dict["GSC"].data)))
            #else:
            #    data_masked = np.ma.array(accum_dict[key].data, mask = np.isnan(accum_dict[key].data))
            
            #time_av_result = np.ma.average(data_masked,axis=0) #average over time dimension
            time_av_result = np.average(accum_dict[key].data,axis=0) #average over time dimension, strict nan corruption
            #time_av_result = np.where(time_av_result==0.,np.nan,time_av_result) #exact zeros are null points
            
            print(key)
            print(time_av_result.shape)
            
            result_dict[key].data.append(time_av_result)
            
        del accum_dict #remove accum dict, we'll set up a new empty one
    
    this_date += td(days=1)

print(result_dict)

#latitudes and longitudes
if use_mod_data and use_sat_data:
    lat_for_nc = new_lat
    lon_for_nc = new_lon
elif use_mod_data:
    lat_for_nc = mod_lat
    lon_for_nc = mod_lon
else:
    lat_for_nc = sat_lat
    lat_for_nc = sat_lon


if option.do_geos_o3_wAK: #add in priors
    result_dict["GC_O3_wAK_wprior"] = val_record("GEOS O3 with OMI AKs inc prior","molec cm-2")
    result_dict["GC_O3_wAK_wprior"].datelist = result_dict["GC_O3_wAK"].datelist
    result_dict["GC_O3_wAK_wprior"].data = np.add(result_dict["GC_O3_wAK"].data,result_dict["prior"].data)
    
    #result_dict["GC_O3_wAK_SUBprior"] = val_record("GEOS O3 with OMI AKs SUB prior","molec cm-2")
    #result_dict["GC_O3_wAK_SUBprior"].datelist = result_dict["GC_O3_wAK"].datelist
    #result_dict["GC_O3_wAK_SUBprior"].data = np.subtract(result_dict["GC_O3_wAK"].data,result_dict["prior"].data)

if option.do_gsc and not option.do_3D: #do bias correction
    print("doing bias correction")

    #fill here??

    
    #print result_dict["GSC_wbiascorr"].data
    
if option.do_gsc and option.do_3D:
    print("Bias correction not done with 3D GSC output")

#once we're done, write to a netcdf file
#prepare NETCDF file
dataset = nc4.Dataset('%s_%s-%s.nc'%(  option.sav_pre,
                                       option.start_date.strftime("%Y%m%d"),
                                       option.end_date.strftime("%Y%m%d"  ))
                                       ,'w', format='NETCDF4') #'w' stands for write


#Double check only those within domain. Make numpy array
lat_for_nc_np = np.array([lat_for_nc[k] <= option.domain[1] and lat_for_nc[k] >= option.domain[0] for k in range(len(lat_for_nc))])
lon_for_nc_np = np.array([lon_for_nc[k] <= option.domain[3] and lon_for_nc[k] >= option.domain[2] for k in range(len(lon_for_nc))])      
#Make dimension    
netcdf_lat = dataset.createDimension('lat', len(lat_for_nc_np))
netcdf_lon = dataset.createDimension('lon', len(lon_for_nc_np))
#co-ordinate variables
latitudes     = dataset.createVariable('lat', np.float32, ('lat',))
latitudes[:]  = lat_for_nc_np
longitudes    = dataset.createVariable('lon', np.float32, ('lon',))
longitudes[:] = lon_for_nc_np
if option.do_3D:
    netcdf_lev    = dataset.createDimension('lev', len(out_levs))
    levels        = dataset.createVariable('lev', np.int32, ('lev',))
    levels[:]     = range(0,len(out_levs)) #integer references

#Time dimension (unlimited)
netcdf_time = dataset.createDimension('time', None)
times = dataset.createVariable('time', np.float64, ('time',))
times.units = 'days since 0001-01-01 00:00:00'
times.calendar = 'gregorian'

first = True

#create the varibales
for key in result_dict:
    if key == "AK":
        continue #don't write out AKs
    
    if first: #for first variable, fill in times       
        for i in range(len(result_dict[key].datelist)):
            times[i] = result_dict[key].datelist[i].toordinal()
        first = False  #don't do this again
    
    #create dataset
    print(key)
    if len(np.array(result_dict[key].data).shape) == 4: #if its a 3D variable
        nc_var = dataset.createVariable(result_dict[key].name, np.float32, ('time','lev','lat','lon'))
        nc_var.units = result_dict[key].unit
        if len(result_dict[key].data[0])==19:
            nc_var[:,:,:,:] = simple_fixed_regrid_19_10(result_dict[key].data) #regrid if on 19 layers
        else:
            nc_var[:,:,:,:] = result_dict[key].data    
    else: #if its a 2D variable
        nc_var = dataset.createVariable(result_dict[key].name, np.float32, ('time','lat','lon'))
        nc_var.units = result_dict[key].unit
        nc_var[:,:,:] = result_dict[key].data
        
        if key == "lossOx":
            print result_dict[key].data        
        






