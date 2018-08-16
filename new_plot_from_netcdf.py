#!/usr/bin/env python

from mpl_toolkits.basemap import Basemap, cm
# requires netcdf4-python (netcdf4-python.googlecode.com)
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
import sys

def frange(x, y, jump):
  while x <= y:
    yield x
    x += jump

force_nest = True
domain = [2.,38.,65.,100]

#dev_from_av = False

plot_fig = True

#plot_scatter = False

cycle_type = "year"
start_date = dt(2014,01,01)
end_date   = dt(2014,12,30)

save_pre = '/geos/u28/scripts/GEOS-Chem_columns/alt_monthly_new/NEW_m_'

if cycle_type == "daily":

    season_list = [start_date + td(days=i) for i in range((end_date-start_date).days+1)]
    #print season_list
    #sys.exit()
elif cycle_type == "monthly":

    season_list = [start_date + td(days=i) for i in range((end_date-start_date).days+1) if (start_date + td(days=i)).day == 1]
elif cycle_type == "year":
    season_list = ["year"]

             #{quickname: [NC_fieldname, full_name,lower_plotbound,upper_plot_bound]
fields_dict = {
               'GC_O3_wAK'     :['GC_O3 with OMI AKs'            ,'GEOS-Chem, with OMI AKs, tropospheric ozone column (no prior)' ,0,   1.0e18],
               'GC_O3_wAK_wpri':['GEOS O3 with OMI AKs inc prior','GEOS-Chem, with OMI AKs, tropspheric ozone column (with prior)',0,   1.0e18],
               #'GC_O3_wAK_SUBpri':['GEOS O3 with OMI AKs SUB prior','GEOS-Chem, with OMI AKs, tropspheric ozone column (with prior subtracted)',0,   1.0e18],
               'MACC_O3'       :['MACC O3'                       ,'MACC tropospheric ozone column'                                ,0,   1.0e18],
               'MACC_O3_wAK'   :['MACC O3 with OMI AKs'          ,'MACC, with OMI AKs, tropospheric ozone column'                 ,0,   1.0e18],
               'prior'         :['prior'                         ,'Prior tropospheric O3 column'                                  ,0,   1.0e18],
               'sat_o3'        :['OMI O3'                        ,'OMI tropospheric ozone column'                                 ,0,   1.0e18],
               'sat_o3_wBC'    :['OMI O3 with bias correction'   ,'OMI tropospheric ozone column, with bias correction'           ,0,   1.0e18],
               'GC_CO'         :['GC_CO'                         ,'GEOS-Chem tropospheric CO column'                              ,0,   1.5e18],
               'GC_CO_GL'      :['GC_CO_GL'                      ,'GEOS-Chem ground-level CO mixing ratio'                        ,0,   4e-7  ],
               'GC_NO'         :['GC_NO'                         ,'GEOS-Chem tropospheric NO column'                              ,0,   1e16  ],
               'GC_NO_GL'      :['GC_NO_GL'                      ,'GEOS-Chem ground-level NO mixing ratio'                        ,0,   5e-9  ],
               'GC_NO2'        :['GC_NO2'                        ,'GEOS-Chem tropospheric NO2 column'                             ,0,   2.5e16],
               'GC_NO2_GL'     :['GC_NO2_GL'                     ,'GEOS-Chem ground-level NO2 mixing ratio'                       ,0,   5e-9  ],
               'GC_NOx'        :['GC_NOx'                        ,'GEOS-Chem tropospheric NOx column'                             ,0,   2.5e16],
               'GC_NOx_GL'     :['GC_NOx_GL'                     ,'GEOS-Chem ground-level NOx mixing ratio'                       ,0,   5e-9  ],
               'GC_CH2O'       :['GC_CH2O'                       ,'GEOS-Chem tropospheric HCHO column'                            ,0,   1.5e16],
               'GC_CH2O_GL'    :['GC_CH2O_GL'                    ,'GEOS-Chem ground-level HCHO mixing ratio'                      ,0,   3.5e-9],
               'GC_O3'         :['GC_O3'                         ,'GEOS-Chem tropospheric O3 column'                              ,0,   1.0e18],
               'GC_O3_GL'      :['GC_O3_GL'                      ,'GEOS-Chem ground-level O3 mixing ratio'                        ,0,   80.e-9]
              }                                   
              
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
    
    #define file
    nc = NetCDFFile('/geos/u28/scripts/GEOS-Chem_columns/alt_monthly_new/NEW_m_%s-%s.nc'%(start_date.strftime('%Y%m%d'),end_date.strftime('%Y%m%d')))

    #spatial stuff
    latcorners = nc.variables['lat'][:]
    loncorners = -nc.variables['lon'][:]
    latres = latcorners[1] - latcorners[0] 
    lonres = -loncorners[1] - -loncorners[0] 

    if force_nest:
        [minlat,maxlat,minlon,maxlon]=[2.,38.,65.,100]
    else:
        minlat = min(latcorners)-0.5*latres
        maxlat = max(latcorners)+0.5*latres
        minlon = min(-loncorners)-0.5*lonres
        maxlon = max(-loncorners)+0.5*lonres
    
    #Now, loop through fields:    
    for field_QN in fields_dict:
        [field_NC,field_full,lower,upper] = fields_dict[field_QN]
        print field_full
        
        plot_var = nc.variables[field_NC]        
        data = plot_var[t][:]
        #mask with same data for now
        mask = plot_var[t][:]
        units = plot_var.units
         
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
        cs = m.pcolormesh(x,y,data_m,vmin=min(clevs),vmax=max(clevs))#,map=plt.cm.jet)
        cbar = m.colorbar(cs,location='bottom',pad="5%")
        cbar.set_label(cbar_text,fontsize=10)
        
        #title
        plt.title("%s - %s"%(field_full,date_text),fontsize=14)
        
        #save
        plt.savefig("%s_%s_%s"%(save_pre,field_QN,date_text))
    
    #clock forward
    t += 1     
                
