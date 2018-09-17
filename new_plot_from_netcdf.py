#!/usr/bin/env python

from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
import sys
from scipy import stats
from itertools import tee, izip
#from numpy.polynomial.polynomial import polyfit

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def load_regiondata(filename,states=True):
    """Reads a csv file which reports the country (and, optionally, the state) on a grid"""
    cs = open(filename,"r")
    #Empty lists to populate
    cs_lat = []
    cs_lon = []
    cs_country = []
    if states:
        cs_state = []   
    for line in cs:
        line = line.strip()
        columns = line.split(",")
        cs_lat.append(float(columns[0]))
        cs_lon.append(float(columns[1]))
        cs_country.append( columns[2] )
        if states:
            cs_state.append( columns[3]    )    
    cs.close() #we can close the file here
    if states:
        return(cs_lat,cs_lon,cs_country,cs_state)
    else:
        return(cs_lat,cs_lon,cs_country)

def region_matcher_fast(cs_lat,cs_lon,cs_country,cs_state,lat,lon,states=True,region="india"):

    """A fast version of region_matcher that only works for country_state.csv"""
    
    num_grid = len(lat)
    if num_grid == 1:
        single = True
    else:
        single = False
    
    #np arrays
    (cs_lat,cs_lon,lat,lon) = (np.array(cs_lat),np.array(cs_lon),np.array([lat]),np.array([lon]))
    
    #get i and j indexes
    
    if region=="india":
        j = np.around(np.divide((lat-2.),0.25))
        j = np.maximum(j,np.zeros_like(j))
        j = np.minimum(j,np.zeros_like(j)+144)
        
        i = np.around(np.divide((lon-65.),0.3125))
        i = np.maximum(i,np.zeros_like(i))
        i = np.minimum(i,np.zeros_like(i)+112)
        ind = np.add(113*j,i)
    elif region=="indonesia":
        j = np.around(np.divide((lat-(-12.)),0.25))
        j = np.maximum(j,np.zeros_like(j))
        j = np.minimum(j,np.zeros_like(j)+77)
        
        i = np.around(np.divide((lon-93.9375),0.3125))
        i = np.maximum(i,np.zeros_like(i))
        i = np.minimum(i,np.zeros_like(i)+154)
        ind = np.add(155*j,i)    
    
    #print len(j)
    #print len(ind)
    #print np.min(ind)
    #print np.max(ind)
    #print len(cs_country)
    
    #print "got indexes"
    country_out = []
    if states:
        state_out = []
    
    for line in range(0,num_grid):
        if line % 1000 == 0:
            #print "Assigning %i of %i" %(line,num_grid)
            pass
        try:    
            country_out.append(cs_country[int(ind[line])])
            if states:
                state_out.append(cs_state[int(ind[line])])
        except IndexError:
            print("line = %i" %line)
            print("ind[line] = %g" %ind[line])
            raise ValueError
    
    if single:
        if states:
            return(country_out[0],state_out[0])
        else:
            return(country_out[0])
    else:
        if states:
            return(country_out,state_out)
        else:
            return(country_out)




def frange(x, y, jump):
  while x <= y:
    yield x
    x += jump

force_nest = True
domain = [2.,38.,65.,100]

#dev_from_av = False

do_plot = True

do_scatter = False

do_prodloss = False

cycle_type = "monthly"
start_date = dt(2014,1,01)
end_date   = dt(2014,12,30)

save_pre = '/geos/u28/scripts/GEOS-Chem_columns/new_BC_month/X1_newBC_month'

if cycle_type == "daily":

    season_list = [start_date + td(days=i) for i in range((end_date-start_date).days+1)]
    #print season_list
    #sys.exit()
elif cycle_type == "monthly":

    season_list = [start_date + td(days=i) for i in range((end_date-start_date).days+1) if (start_date + td(days=i)).day == 1]
elif cycle_type == "year":
    season_list = ["year"]

max_o3 = 1.0e18

do_country_mask = False
country_statefile = "country_state.csv"


             #{quickname: [NC_fieldname, full_name,lower_plotbound,upper_plot_bound]
fields_dict = {
               #'GC_O3_wAK'     :['GC_O3 with OMI AKs'            ,'GEOS-Chem, with OMI AKs, tropospheric ozone column (no prior)' ,0,   max_o3],
               'GC_O3_wAK_wpri':['GEOS O3 with OMI AKs inc prior','GEOS-Chem, with OMI AKs, tropspheric ozone column (with prior)',0.0e18,   max_o3],
               #'GC_O3_wAK_SUBpri':['GEOS O3 with OMI AKs SUB prior','GEOS-Chem, with OMI AKs, tropspheric ozone column (with prior subtracted)',0,   max_o3],
               'MACC_O3'       :['MACC O3'                       ,'MACC tropospheric ozone column'                                ,0,   max_o3],
               #'MACC_O3_wAK'   :['MACC O3 with OMI AKs'          ,'MACC, with OMI AKs, tropospheric ozone column'                 ,0,   max_o3],
               #'prior'         :['prior'                         ,'Prior tropospheric O3 column'                                  ,0,   max_o3],
               'sat_o3'        :['OMI O3'                        ,'OMI tropospheric ozone column'                                 ,0,   max_o3],
               'sat_o3_wBC'    :['OMI O3 with bias correction'   ,'OMI tropospheric ozone column, with bias correction'           ,0.0e18,   max_o3],
               'sat_o3_wBC_old'    :['OMI O3 with OLD bias correction'   ,'OMI tropospheric ozone column, with OLD bias correction'           ,0.0e18,   max_o3],
               #'GC_CO'         :['GC_CO'                         ,'GEOS-Chem tropospheric CO column'                              ,0,   1.5e18],
               #'GC_CO_GL'      :['GC_CO_GL'                      ,'GEOS-Chem ground-level CO mixing ratio'                        ,0,   4e-7  ],
               #'GC_NO'         :['GC_NO'                         ,'GEOS-Chem tropospheric NO column'                              ,0,   1e16  ],
               #'GC_NO_GL'      :['GC_NO_GL'                      ,'GEOS-Chem ground-level NO mixing ratio'                        ,0,   5e-9  ],
               #'GC_NO2'        :['GC_NO2'                        ,'GEOS-Chem tropospheric NO2 column'                             ,0,   2.5e16],
               #'GC_NO2_GL'     :['GC_NO2_GL'                     ,'GEOS-Chem ground-level NO2 mixing ratio'                       ,0,   5e-9  ],
               #'GC_NOx'        :['GC_NOx'                        ,'GEOS-Chem tropospheric NOx column'                             ,0,   2.5e16],
               #'GC_NOx_GL'     :['GC_NOx_GL'                     ,'GEOS-Chem ground-level NOx mixing ratio'                       ,0,   5e-9  ],
               #'GC_CH2O'       :['GC_CH2O'                       ,'GEOS-Chem tropospheric HCHO column'                            ,0,   1.5e16],
               #'GC_CH2O_GL'    :['GC_CH2O_GL'                    ,'GEOS-Chem ground-level HCHO mixing ratio'                      ,0,   2.5e-9],
               'GC_O3'         :['GC_O3'                         ,'GEOS-Chem tropospheric O3 column'                              ,0.0e18,   max_o3],
               #'GC_O3_GL'      :['GC_O3_GL'                      ,'GEOS-Chem ground-level O3 mixing ratio'                        ,30.e-9,   60.e-9],
               'bias_corr'      :['bias correction'               ,'Bias correction applied'                                       ,0,   3e17]
               #'prod_Ox'       :['prod_Ox'                       ,'Ox production rate'                                            ,0,   2.5e12],
               #'loss_Ox'       :['loss_Ox'                       ,'Ox loss rate'                                                  ,0,   2.5e12]
              }                                   

(csv_lat,csv_lon,csv_country,csv_state) = \
            load_regiondata(country_statefile,states=True) #get countries for points

#define file
nc = NetCDFFile('/geos/u28/scripts/GEOS-Chem_columns/NewBC_x1_monthly__%s-%s.nc'%(start_date.strftime('%Y%m%d'),end_date.strftime('%Y%m%d')))
#nc = NetCDFFile('/geos/u28/scripts/GEOS-Chem_columns/PL_goodsat_x1_new_2D_%s-%s.nc'%(start_date.strftime('%Y%m%d'),end_date.strftime('%Y%m%d')))


#spatial stuff
latcorners = nc.variables['lat'][:]
loncorners = -nc.variables['lon'][:]
latres = latcorners[1] - latcorners[0] 
lonres = -loncorners[1] - -loncorners[0] 

#print latcorners
#print loncorners
#print latres
#print lonres

if force_nest:
    [minlat,maxlat,minlon,maxlon]=[2.,38.,65.,100]
else:
    minlat = min(latcorners)-0.5*latres
    maxlat = max(latcorners)+0.5*latres
    minlon = min(-loncorners)-0.5*lonres
    maxlon = max(-loncorners)+0.5*lonres

#get OMI mask

plot_var = nc.variables['OMI O3'] 
out_india_mask = np.zeros(np.array(plot_var[0][:]).shape,dtype=bool)
for i in range(len(out_india_mask)):
    this_lat = minlat + i*0.75
    for j in range(len(out_india_mask[0])):                
        this_lon = minlon + j*0.75
        (this_country,this_state) = region_matcher_fast(csv_lat,csv_lon,csv_country,csv_state,
                           [this_lat],[this_lon],region="india")
        #print this_lat,this_lon,this_country
        if do_country_mask:
            if this_country in ["India","Bangladesh"]:
                out_india_mask[i][j] = False
            else:
                out_india_mask[i][j] = True
        else:
            out_india_mask[i][j] = False
top_mask = out_india_mask
for t in range(len(season_list)):   
    zero_or_nan_mask = np.logical_or(np.isnan(plot_var[t][:]),plot_var[t][:] == 0.)   
    top_mask = np.logical_or(zero_or_nan_mask,top_mask) #update mask


              
for t in range(len(season_list)):
    print t
    
    #text for filename and headers        
    if season_list[t] == "JFM":
        date_text = '20140101-20140331'
    elif season_list[t] == "AMJ":
        date_text = '20140401-20140630'
    elif season_list[t] == "JAS":
        date_text = '20140701-20140930'
    elif season_list[t] == "OND":
        date_text = '20141001-20141230'
    elif season_list[t] == "year":
        date_text = start_date.strftime('%Y%m%d')+"-"+end_date.strftime('%Y%m%d')
    else: #month or daily cycle
        if cycle_type == "daily":
            date_text = season_list[t].strftime('%Y%m%d')
        elif cycle_type == "monthly":
            date_text = season_list[t].strftime('%Y%m')
        else: #we shouldn't get here
            raise IOError("failure to set date_text")

    print date_text
       
    #Now, loop through fields:    
    for field_QN in fields_dict:
        [field_NC,field_full,lower,upper] = fields_dict[field_QN]
                
        plot_var = nc.variables[field_NC]        
        data = plot_var[t][:]
        #print "data shape"
        #print np.array(data).shape
        
        #out_india_mask = np.zeros(np.array(data).shape,dtype=bool)
        #for i in range(len(out_india_mask)):
        #    this_lat = minlat + i*0.75
        #    for j in range(len(out_india_mask[0])):                
        #        this_lon = minlon + j*0.75
        #        (this_country,this_state) = region_matcher_fast(csv_lat,csv_lon,csv_country,csv_state,
        #                           [this_lat],[this_lon],region="india")
        #        #print this_lat,this_lon,this_country
        #        if this_country in ["India","Bangladesh"]:
        #            out_india_mask[i][j] = False
        #        else:
        #            out_india_mask[i][j] = True
            
        
        #mask with same data for now
        mask = plot_var[t][:]
        units = plot_var.units
        #mask
        data_m = np.ma.masked_where(top_mask,data)
        
        print "%s : average : %g" %(field_full,np.nanmean(data_m.flatten()))
        
        
        
        if do_plot: 
            #basic draw map stuff
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_axes([0.1,0.1,0.8,0.8])
            # create Mercator Projection Basemap instance.

            m = Basemap(projection='merc',llcrnrlat=minlat-1.75,urcrnrlat=maxlat-1.75,\
                        llcrnrlon=minlon-0.75,urcrnrlon=maxlon-0.75,lat_ts=20,resolution='i')

            # draw coastlines, state and country boundaries, edge of map.
            m.drawcoastlines()
            #m.drawstates()
            m.drawcountries()
            # draw parallels.
            parallels = np.arange(-90.,90,5.)
            m.drawparallels(parallels,labels=[1,0,0,0],fontsize=16)
            # draw meridians
            meridians = np.arange(-180.,180.,5.)
            m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=16)
            ny = data.shape[0]; nx = data.shape[1]
            lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
            x, y = m(lons, lats) # compute map proj coordinates.
            
            #colour bar        
            #(lower,upper) = (np.nanmin(np.array(data).flatten()),np.nanmax(np.array(data).flatten()))
            step  = (upper-lower)/20.
            
            clevs = list(frange(lower,upper,step))
            cbar_text = units       
            
            
            #main fig
            #colormap
            if field_QN == "prod_Ox":
                cmap = plt.cm.BuPu
            elif field_QN == "loss_Ox":
                cmap = plt.cm.YlGn
            elif field_QN.endswith("_GL"):
                cmap = plt.cm.inferno
            else:
                cmap = plt.cm.nipy_spectral
            #plot it
            cs = m.pcolormesh(x,y,data_m,vmin=min(clevs),vmax=max(clevs),cmap=cmap)
            cbar = m.colorbar(cs,location='bottom',pad="5%")
            cbar.set_label(cbar_text,fontsize=10)
            
            #title
            plt.title("%s - %s"%(field_full,date_text),fontsize=14)
            
            #save
            plt.savefig("%s_%s_%s"%(save_pre,field_QN,date_text))
            
            plt.close()
        
        if field_QN == "prod_Ox":
            prod_Ox_data = data
            prod_Ox_mask = mask
        if field_QN == "loss_Ox":
            loss_Ox_data = data
            loss_Ox_mask = mask    
    
    if do_scatter:
        
        for xfield_QN, yfield_QN in pairwise(fields_dict):
        
            [xfield_NC,xfield_full,xlower,xupper] = fields_dict[xfield_QN]
            [yfield_NC,yfield_full,ylower,yupper] = fields_dict[yfield_QN]
                
            xaxis_var = nc.variables[xfield_NC]
            yaxis_var = nc.variables[yfield_NC]        
            xdata = np.array(xaxis_var[t][:]).flatten()
            ydata = np.array(yaxis_var[t][:]).flatten()
            
            fig = plt.figure(figsize=(8,8))
            plt.scatter(xdata,ydata,alpha=0.2)
            plt.title("%s"%(date_text),fontsize=14)
            plt.xlabel("%s / %s" %(xfield_full,xaxis_var.units))
            plt.ylabel("%s / %s" %(yfield_full,yaxis_var.units))
            plt.axis([xlower, xupper, ylower, yupper])
            plt.grid(True)
            
            
            #for stats, strip out wherever either is nan
            xdata_clean = xdata[np.logical_not(np.isnan(np.multiply(xdata,ydata)))]
            ydata_clean = ydata[np.logical_not(np.isnan(np.multiply(xdata,ydata)))]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(xdata_clean, ydata_clean)
            
            print "m = %g , c = %g , rsq = %g "%(slope,intercept,r_value*r_value)
            
                        
            plt.plot([xlower,xupper], [intercept + slope * xlower,intercept + slope * xupper], 'r-')
            
            plt.annotate('y = %.2f x + %.2E , r2 = %.2f'%(slope,intercept,r_value*r_value), (0,0), (0, -45), xycoords='axes fraction', textcoords='offset points', va='top')
            
            plt.savefig("%s_Scatter_X-%s_Y-%s_%s"%(save_pre,xfield_QN,yfield_QN,date_text))
            plt.close()
    
    
    if do_prodloss: #prodloss
        #=====do net Ox===========
        field_QN = "net_Ox"
        [field_NC,field_full,lower,upper] = ["net_Ox","Net chemical Ox production/loss",-2.5e12,2.5e12]
        data = prod_Ox_data - loss_Ox_data
        mask = prod_Ox_mask
        units = "molec/cm3/s"
        
        #basic draw map stuff
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        # create Mercator Projection Basemap instance.

        m = Basemap(projection='merc',llcrnrlat=minlat,urcrnrlat=maxlat,\
                    llcrnrlon=minlon,urcrnrlon=maxlon,lat_ts=20,resolution='i')

        # draw coastlines, state and country boundaries, edge of map.
        m.drawcoastlines()
        #m.drawstates()
        m.drawcountries()
        # draw parallels.
        parallels = np.arange(-90.,90,5.)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=16)
        # draw meridians
        meridians = np.arange(-180.,180.,5.)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=16)
        ny = data.shape[0]; nx = data.shape[1]
        lons, lats = m.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
        x, y = m(lons, lats) # compute map proj coordinates.
        
        #colour bar        
        #(lower,upper) = (np.nanmin(np.array(data).flatten()),np.nanmax(np.array(data).flatten()))
        step  = (upper-lower)/20.
        
        clevs = list(frange(lower,upper,step))
        cbar_text = units       
        #mask
        data_m = np.ma.masked_where(np.isnan(mask),data)
        
        #main fig
        cs = m.pcolormesh(x,y,data_m,vmin=min(clevs),vmax=max(clevs),cmap=plt.cm.bwr)
        cbar = m.colorbar(cs,location='bottom',pad="5%")
        cbar.set_label(cbar_text,fontsize=10)
        
        #title
        plt.title("%s - %s"%(field_full,date_text),fontsize=14)
        
        #save
        plt.savefig("%s_%s_%s"%(save_pre,field_QN,date_text))
        
        plt.close()
    
    #===================
    
    #clock forward
    t += 1     
                
