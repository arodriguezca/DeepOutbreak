#%%%

# Grouped Bar Chart
# Nikhil Muralidhar <nik90@vt.edu>
# Fri 6/5/2020 12:27 PM
#Function
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_bar_chart(error_dict,xticklabels,figname,colors,legend_outside=False):
    N=len(error_dict[list(error_dict.keys())[0]])
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.05         # the width of the bars
   
    bars=list()
    k=2
    for idx,model_name in enumerate(error_dict.keys()):
        print(model_name,len(error_dict[model_name]))
        _p = ax.bar(ind+((idx-k)*width), error_dict[model_name], width, bottom=0,color=colors[idx])
        bars.append(_p)

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # ax.set_xticks(ind + width / 2)
   
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(20)
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small')
        #tick.label.set_rotation('vertical')
   
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
   
    # ax.set_xticklabels(xticklabels)
   
    # ax.set_xlabel("Region",fontsize=26)
    ax.set_ylabel('RMSE',fontsize=26)
    # ax.legend(bars,error_dict.keys(),fontsize=18,ncol=2,loc=2)
    ax.legend(bars,error_dict.keys(),fontsize=18,ncol=2,loc='lower center')
    fig.tight_layout()
    if legend_outside:
        ax.legend(bars,error_dict.keys(),fontsize=18,ncol=2,bbox_to_anchor=(0.1, 1.02))
        ax.set_ylim(top=2.)
        # plt.figlegend()
    if figname:
        fig.savefig(figname,dpi=300)
    #ax.legend((p1[0], p2[0]), ('Men', 'Women'))
    #ax.yaxis.set_units(inch)
   # ax.autoscale_view()

#%%
# How to Call
fig_outputdir="./"

#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X
#data_ourmethod=[0.7983, 2.4382, 1.1294, 1.4995, 1.2873, 2.5763, 1.9178, 1.1856, 1.6165, 0.993, 1.2214]
data_ourmethod_lap=[1.523508978]
data_global_recurrent=[1.88673271] 
data_gru_baseline=[1.805339969]
data_epideep=[2.776474022]
errs_dict=OrderedDict({'DS1':data_ourmethod_lap,
                       'DS2':data_global_recurrent,
                       'DS3':data_gru_baseline,
                       'DS4':data_epideep})

colors=['r','orange','b','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','National']

figname=fig_outputdir+"data_ablation.png"

plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


#%%
# for model ablation
fig_outputdir="./figures/"
data_ourmethod_lap=[1.736005234]
data_global_recurrent=[10.6477247] 
data_gru_baseline=[1.838689042]
data_epideep=[1.811749546]
errs_dict=OrderedDict({'CALI-NET':data_ourmethod_lap,
                       '-GRU':data_global_recurrent,
                       '-DAE':data_gru_baseline,
                       '-LAP':data_epideep})

colors=['r','orange','b','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','National']

figname=fig_outputdir+"model_ablation.png"

plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


#%%
# for model ablation
fig_outputdir="./figures/"
data_ourmethod_lap=[1.736005234]
kinsa=[1.885402009] 
linelist=[2.007767718]
social=[1.862400398]
testing=[1.730176188]
errs_dict=OrderedDict({'DS1':linelist,
                       'DS2':testing,
                       'DS3':kinsa,
                       'DS4':social})

colors=['r','orange','b','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','National']

figname=fig_outputdir+"real_data_ablation.png"

plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors,False)
#%%
# for laplacian
fig_outputdir="./figures/"
one=[1.819509039]
two=[1.791183097]
three=[1.82749144] 
errs_dict=OrderedDict({'0.001':one,
                       '0.01':two,
                       '0.1':three})

colors=['r','orange','b','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','National']

figname=fig_outputdir+"laplacian.png"

plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors,True)


# for KD
fig_outputdir="./figures/"
one=[1.885886686]
two=[1.942237212]
three=[1.861726594] 
errs_dict=OrderedDict({'0.1':one,
                       '0.3':two,
                       '0.5':three})

colors=['r','orange','b','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','National']

figname=fig_outputdir+"KD.png"

plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors,True)


# for region recon
fig_outputdir="./figures/"
one=[1.778847146]
two=[1.791183097]
three=[1.782167492] 
errs_dict=OrderedDict({'0.001':one,
                       '0.01':two,
                       '0.1':three})

colors=['r','orange','b','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','National']

figname=fig_outputdir+"region_recon.png"

plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors,True)






#%%
# new AAAI image
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


def plot_grouped_bar_chart(error_dict,xticklabels,figname,colors):
    N=len(error_dict[list(error_dict.keys())[0]])
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.26         # the width of the bars
    
    bars=list()
    k=1
    for idx,model_name in enumerate(error_dict.keys()):
        print(model_name,len(error_dict[model_name]))
        _p = ax.bar(ind+((idx-k)*width), error_dict[model_name], width, bottom=0,color=colors[idx])
        bars.append(_p)

    ax.set_xticks(ind + width / 2)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small') 
        # tick.label.set_rotation('vertical')
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12) #20
    
    ax.set_xticklabels(xticklabels)
    
    ax.set_xlabel("Region",fontsize=24)
    ax.set_ylabel('RMSE',fontsize=24)
    ax.set_ylim((0,7))
    ax.legend(bars,error_dict.keys(),fontsize=18)
    fig.tight_layout()
    fig.savefig(figname,dpi=300)

# How to Call
fig_outputdir="./figures/"


#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X
calinet=[0.571528,1.952536,0.556232,0.326371,0.215556,0.769148,0.487863,0.945001,0.419612,0.705652,0.302784]
# epideep=[3.45380607,6.860763847,3.321712803,3.558746638,2.334710963,4.543208084,4.222881299,4.33097838,1.258114131,3.727908219,3.323004558] 
eb=[3.403624,5.44888,3.198409,3.390738,2.144143,4.021613,4.270399,4.179896,1.341486,3.692458,3.218463]
# var=[1.489018733,4.136260237,1.406122542,1.661253317,1.25119278,2.09304434,2.043638838,1.971990481,0.9989388183,1.250600353,1.303988309]
delta_density1=[1.359097,2.264011,0.99965,1.017109,0.479968,0.716494,0.553736,1.310855,0.433855,1.057106,0.875026]
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','US']
xticklabels=['1','2','3','4','5','6','7','8','9','10','US']
#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X

# del calinet[5:7]
# del epideep[5:7]
# del eb[5:7]
# del var[5:7]
# del delta_density1[5:7]
# del xticklabels[5:7]

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'Emp. Bayes':eb,
                    #    'VAR':var,
                       'DeltaDensity':delta_density1})

# colors=['r','y','purple','green','dimgrey']
colors=['r','purple','dimgrey']
figname=fig_outputdir+"adaptation1_all_next1EW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)























#%%

# new AAAI image
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np


def plot_grouped_bar_chart(error_dict,xticklabels,figname,colors):
    N=len(error_dict[list(error_dict.keys())[0]])
    fig, ax = plt.subplots(1,1,figsize=(10,5))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.3         # the width of the bars
    
    bars=list()
    k=1
    for idx,model_name in enumerate(error_dict.keys()):
        print(model_name,len(error_dict[model_name]))
        _p = ax.bar(ind+((idx-k)*width), error_dict[model_name], width, bottom=0,color=colors[idx])
        bars.append(_p)

    ax.set_xticks(ind + width / 2)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small') 
        # tick.label.set_rotation('vertical')
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12) #20
    
    ax.set_xticklabels(xticklabels)
    
    ax.set_xlabel("Region",fontsize=24)
    ax.set_ylabel('RMSE',fontsize=24)
    ax.set_ylim((0,7))
    ax.legend(bars,error_dict.keys(),fontsize=18)
    fig.tight_layout()
    fig.savefig(figname,dpi=300)

# How to Call
fig_outputdir="./figures/"

# NIPS results, week 11

#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X
calinet=[0.571528,1.952536,0.556232,0.326371,0.215556,0.769148,0.487863,0.945001,0.419612,0.705652,0.302784]
epideep=[3.45380607,6.860763847,3.321712803,3.558746638,2.334710963,4.543208084,4.222881299,4.33097838,1.258114131,3.727908219,3.323004558] 
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','US']
xticklabels=['1','2','3','4','5','6','7','8','9','10','US']
errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep})

colors=['r','y','purple','green','dimgrey']
figname=fig_outputdir+"adaptation1_next1EW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


calinet=[0.805289,1.233803,0.753546,1.009729,0.966822,2.028373,1.564499,0.903377,1.197437,0.858074,0.650328]
epideep=[1.209050965,5.124674894,1.126778795,0.8787012835,0.6775200844,1.045728224,0.5239921574,1.077067619,0.3409415724,2.303562758,1.044208294]
errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep})
colors=['r','y','purple','green','dimgrey']
figname=fig_outputdir+"adaptation2_next1EW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)







# %%
# NIPS results, week 11

#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X
calinet=[0.9871683356,4.463213961,1.083768029,1.224590879,1.692728255,2.464275938,2.38292012,1.312742414,1.5216422,1.037440126,0.9975127308]
epideep=[3.519445786,7.654343519,2.780441019,2.874513911,1.821010638,3.714527899,3.241868142,3.534956533,1.053837081,2.690543293,2.933681613] 
eb=[3.46771771,7.77325557,3.020880515,2.928333559,1.767188015,3.503780219,3.126201378,3.484963843,1.272070105,3.048840474,3.033808267]
var=[1.489018733,4.136260237,1.406122542,1.661253317,1.25119278,2.09304434,2.043638838,1.971990481,0.9989388183,1.250600353,1.303988309]
delta_density1=[2.184850918,4.897755621,1.604185136,1.735143298,0.8876371039,1.33488448,1.664579285,1.826423217,0.751986316,1.443425342,1.281996229]

xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','US']
del calinet[5:7]
del epideep[5:7]
del eb[5:7]
del var[5:7]
del delta_density1[5:7]
del xticklabels[5:7]

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'Emp. Bayes':eb,
                       'VAR':var,
                       'DeltaDensity':delta_density1})

colors=['r','y','purple','green','dimgrey']
figname=fig_outputdir+"adaptation1_neuripsEW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'DeltaDensity':delta_density1})
figname=fig_outputdir+"adaptation2_neuripsEW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


# NIPS results, week 12

calinet=[1.180989183, 3.653108474,1.576025783,1.791622429,2.107139039,3.239039129,2.834668724,1.715940529,1.985497789,1.46256617, 1.497075991]
epideep=[3.332621424,7.470510365,2.514075555,2.590920444,1.60900671,3.305223933,2.763320073,3.103214795,0.9734686425,2.693037422,2.448336095]
eb=[3.281733335,7.630477973,2.70798053,2.589086153,1.52935036,3.080540595,2.641598208,3.045612577,1.102893906,2.717367453,2.752960086]
var=[2.228989304,3.966975198,2.024075205,2.219855273,1.841075156,3.472246843,2.733562738,2.408927603,1.605584514,1.161510441,2.004992028]
delta_density1=[1.891311203,3.946529637,1.819122715,1.953883171,1.117892513,1.807334906,1.934927533,2.009676164,0.9933944965,1.651792314,1.500785943]
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','US']
del calinet[5:7]
del epideep[5:7]
del eb[5:7]
del var[5:7]
del delta_density1[5:7]
del xticklabels[5:7]
errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'Emp. Bayes':eb,
                       'VAR':var,
                       'DeltaDensity':delta_density1})

figname=fig_outputdir+"adaptation1_neuripsEW12.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'DeltaDensity':delta_density1})
figname=fig_outputdir+"adaptation2_neuripsEW12.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


# %%
# new data

#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X
calinet=[0.7890853493,3.893644564,1.19239758,1.901183366,1.448225381,2.724998104,2.558591695,1.762734941,0.850209423,1.303002182,1.201182271]
epideep=[3.519445786,7.654343519,2.780441019,2.874513911,1.821010638,3.714527899,3.241868142,3.534956533,1.053837081,2.690543293,2.933681613] 
delta_density1=[2.184850918,4.897755621,1.604185136,1.735143298,0.8876371039,1.33488448,1.664579285,1.826423217,0.751986316,1.443425342,1.281996229]

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep})

colors=['r','y','dimgrey','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','US']
figname=fig_outputdir+"adaptation1_newEW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'DeltaDensity':delta_density1})
figname=fig_outputdir+"adaptation2_newEW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


# NIPS results, week 12

calinet=[1.110628903,3.360594049,1.577021468,2.442671407,1.823617364,3.381915885,3.037705768,2.248262047,1.196606836,1.517460924,1.687498826]
epideep=[3.332621424,7.470510365,2.514075555,2.590920444,1.60900671,3.305223933,2.763320073,3.103214795,0.9734686425,2.693037422,2.448336095]
delta_density1=[1.891311203,3.946529637,1.819122715,1.953883171,1.117892513,1.807334906,1.934927533,2.009676164,0.9933944965,1.651792314,1.500785943]

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep})

figname=fig_outputdir+"adaptation1_newEW12.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'DeltaDensity':delta_density1})
figname=fig_outputdir+"adaptation2_newEW12.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)

# %%
# new data + mobility

#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X
calinet=[0.8462988937,3.511720612,1.579174193,2.419818089,1.991793137,4.05428446,3.344601845,1.928805007,2.041059756,1.737297753,1.546763168]
epideep=[3.519445786,7.654343519,2.780441019,2.874513911,1.821010638,3.714527899,3.241868142,3.534956533,1.053837081,2.690543293,2.933681613] 
delta_density1=[2.184850918,4.897755621,1.604185136,1.735143298,0.8876371039,1.33488448,1.664579285,1.826423217,0.751986316,1.443425342,1.281996229]


errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep})

colors=['r','y','dimgrey','y','green','purple']
xticklabels=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','US']
figname=fig_outputdir+"adaptation1_newMobEW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'DeltaDensity':delta_density1})
figname=fig_outputdir+"adaptation2_newMobEW11.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)


# NIPS results, week 12

calinet=[1.095000137,3.097952737,2.038771605,2.842005885,2.406506369,4.480258193,3.613016807,2.294099141,2.388388847,2.17238499,1.950244644]
epideep=[3.332621424,7.470510365,2.514075555,2.590920444,1.60900671,3.305223933,2.763320073,3.103214795,0.9734686425,2.693037422,2.448336095]
delta_density1=[1.891311203,3.946529637,1.819122715,1.953883171,1.117892513,1.807334906,1.934927533,2.009676164,0.9933944965,1.651792314,1.500785943]

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep})

figname=fig_outputdir+"adaptation1_newMobEW12.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)

errs_dict=OrderedDict({'CALI-NET':calinet,
                       'EpiDeep':epideep,
                       'DeltaDensity':delta_density1})
figname=fig_outputdir+"adaptation2_newMobEW12.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors)

































# %%


# new AAAI image
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
# plt.rc('text', usetex=True)
# plt.rc('text.latex')
# matplotlib.verbose.level = 'debug-annoying'



def plot_grouped_bar_chart(error_dict,xticklabels,figname,colors,metric,max_):
    N=len(error_dict[list(error_dict.keys())[0]])
    fig, ax = plt.subplots(1,1,figsize=(5,5))

    ind = np.arange(N)    # the x locations for the groups
    width = 0.26         # the width of the bars
    
    bars=list()
    k=1
    for idx,model_name in enumerate(error_dict.keys()):
        print(model_name,len(error_dict[model_name]))
        _p = ax.bar(ind+((idx-k)*width), error_dict[model_name], width, bottom=0,color=colors[idx])
        bars.append(_p)

    ax.set_xticks(ind + width / 2)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
        # specify integer or one of preset strings, e.g.
        #tick.label.set_fontsize('x-small') 
        # tick.label.set_rotation('vertical')
    
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12) #20
    
    ax.set_xticklabels(xticklabels,rotation=0)
    # ax.spines['right'].set_color('none')

    # ax.spines['top'].set_color('none')

    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['left'].set_position(('data', 0.5))
    # ax.set_xlabel("Region",fontsize=24)
    ax.set_ylabel(metric,fontsize=24)
    ax.set_ylim((0,max_))
    ax.legend(bars,error_dict.keys(),fontsize=16)
    fig.tight_layout()
    fig.savefig(figname,dpi=300)

# How to Call
fig_outputdir="./figures/"


#R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,X
calinet=[1527.361176,2114.503540]
eb=[1696.05437, 2085.15715]
metric=r'$\Gamma{\alpha}$'
max_=3000
xticklabels=['1 wk ahead\n inc death', '2 wk ahead\n inc death']
errs_dict=OrderedDict({'DeepCOVID':calinet,
                       'COVID Hub Ensemble':eb})

# colors=['r','y','purple','green','dimgrey']
colors=['b','r','dimgrey']
figname=fig_outputdir+"interval_70_US.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors,metric,max_)



# %%
calinet=[0.1050215,0.13491393]
eb=[0.093440, 0.1077349]
xticklabels=['1 wk ahead\n inc death', '2 wk ahead\n inc death']
errs_dict=OrderedDict({'DeepCOVID':calinet,
                       'COVID Hub Ensemble':eb})
metric='MAPE'
max_=0.18
# colors=['r','y','purple','green','dimgrey']
colors=['b','r','dimgrey']
figname=fig_outputdir+"mape_US.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors,metric,max_)




# %%
calinet=[0.172487, 0.3568993,0.444791,0.218765]
eb=[0.165709, 0.3977215, 0.82171890,0.1609189]
xticklabels=['US','TX', 'VT','CA']
errs_dict=OrderedDict({'DeepCOVID':calinet,
                       'COVID Hub Ensemble':eb})
metric='MAPE'
max_=0.9
# colors=['r','y','purple','green','dimgrey']
colors=['b','r','dimgrey']
figname=fig_outputdir+"mape_states.png"
plot_grouped_bar_chart(errs_dict,xticklabels,figname,colors,metric,max_)