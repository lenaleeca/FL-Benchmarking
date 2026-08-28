import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FLc="#6A61C6"; CENTc="#C07D2A"
# (task,site): {AUC:{WN:(fl,cent), LSO:(fl,cent)}, BRIER:{...}}  each entry (val,lo,hi)
D={
("Sepsis","Site 1"):{"AUC":{"WN":((0.001,-0.002,0.004),(0.010,0.006,0.015)),"LSO":((0.001,-0.003,0.006),(0.000,-0.004,0.005))},
                      "BRIER":{"WN":((-0.002,-0.005,0.001),(-0.010,-0.014,-0.006)),"LSO":((0.000,-0.005,0.004),(0.000,-0.004,0.004))}},
("Sepsis","Site 2"):{"AUC":{"WN":((0.002,-0.002,0.007),(-0.037,-0.061,-0.018)),"LSO":((0.004,-0.001,0.009),(-0.060,-0.087,-0.036))},
                      "BRIER":{"WN":((0.000,-0.006,0.007),(0.172,0.154,0.186)),"LSO":((-0.004,-0.009,0.002),(0.172,0.154,0.186))}},
("Sepsis","Site 3"):{"AUC":{"WN":((0.004,-0.001,0.010),(-0.047,-0.073,-0.026)),"LSO":((0.002,-0.005,0.010),(-0.106,-0.137,-0.076))},
                      "BRIER":{"WN":((-0.005,-0.009,-0.001),(0.201,0.188,0.213)),"LSO":((0.072,0.060,0.083),(0.200,0.187,0.212))}},
("Sepsis","Site 4"):{"AUC":{"WN":((0.018,0.005,0.035),(-0.152,-0.227,-0.080)),"LSO":((-0.077,-0.137,-0.019),(-0.166,-0.236,-0.092))},
                      "BRIER":{"WN":((-0.020,-0.035,-0.007),(0.242,0.209,0.271)),"LSO":((0.240,0.207,0.270),(0.242,0.209,0.271))}},
("AMI","Site 1"):{"AUC":{"WN":((0.000,-0.005,0.006),(-0.010,-0.025,0.004)),"LSO":((-0.072,-0.100,-0.047),(-0.048,-0.072,-0.027))},
                   "BRIER":{"WN":((0.001,-0.004,0.006),(0.085,0.070,0.100)),"LSO":((0.116,0.101,0.128),(0.095,0.078,0.109))}},
("AMI","Site 2"):{"AUC":{"WN":((-0.004,-0.031,0.021),(0.008,-0.031,0.047)),"LSO":((-0.056,-0.114,0.001),(0.015,-0.018,0.049))},
                   "BRIER":{"WN":((-0.006,-0.017,0.005),(0.077,0.053,0.097)),"LSO":((0.149,0.123,0.174),(0.074,0.052,0.094))}},
("AMI","Site 3"):{"AUC":{"WN":((-0.029,-0.090,0.028),(-0.120,-0.240,-0.014)),"LSO":((-0.114,-0.262,0.012),(-0.148,-0.269,-0.035))},
                   "BRIER":{"WN":((-0.004,-0.023,0.018),(0.060,0.011,0.109)),"LSO":((0.030,-0.001,0.062),(0.050,0.010,0.094))}},
("AMI","Site 4"):{"AUC":{"WN":((0.003,-0.026,0.031),(-0.048,-0.093,-0.004)),"LSO":((-0.109,-0.194,-0.029),(-0.010,-0.047,0.027))},
                   "BRIER":{"WN":((0.000,-0.017,0.017),(0.039,0.012,0.065)),"LSO":((0.049,0.008,0.088),(0.045,0.018,0.071))}},
("AMI","Site 5"):{"AUC":{"WN":((-0.017,-0.040,0.005),(-0.038,-0.082,0.000)),"LSO":((-0.084,-0.145,-0.029),(-0.041,-0.082,-0.002))},
                   "BRIER":{"WN":((0.004,-0.009,0.016),(0.116,0.084,0.149)),"LSO":((0.092,0.061,0.125),(0.110,0.078,0.144))}},
("AMI","Site 6"):{"AUC":{"WN":((0.002,-0.029,0.032),(0.012,-0.019,0.041)),"LSO":((-0.088,-0.165,-0.016),(-0.002,-0.035,0.032))},
                   "BRIER":{"WN":((-0.007,-0.026,0.010),(-0.006,-0.018,0.006)),"LSO":((0.055,0.027,0.086),(0.011,-0.005,0.027))}},
("Diabetes","Site A"):{"AUC":{"WN":((0.002,-0.002,0.006),(0.015,0.010,0.020)),"LSO":((-0.003,-0.009,0.003),(-0.001,-0.006,0.005))},
                        "BRIER":{"WN":((-0.008,-0.011,-0.005),(-0.012,-0.014,-0.009)),"LSO":((0.000,-0.003,0.003),(0.004,0.001,0.007))}},
("Diabetes","Site B"):{"AUC":{"WN":((0.001,-0.006,0.008),(0.017,0.008,0.026)),"LSO":((-0.006,-0.016,0.004),(-0.006,-0.017,0.005))},
                        "BRIER":{"WN":((0.001,-0.002,0.004),(-0.003,-0.007,0.002)),"LSO":((0.011,0.004,0.017),(0.012,0.005,0.018))}},
("Diabetes","Site C"):{"AUC":{"WN":((-0.001,-0.008,0.006),(0.013,0.005,0.021)),"LSO":((-0.003,-0.013,0.006),(-0.002,-0.012,0.007))},
                        "BRIER":{"WN":((0.000,-0.003,0.003),(-0.005,-0.009,-0.001)),"LSO":((0.003,-0.001,0.008),(0.003,-0.001,0.008))}},
}
order=[("Sepsis",["Site 1","Site 2","Site 3","Site 4"]),("AMI",["Site 1","Site 2","Site 3","Site 4","Site 5","Site 6"]),("Diabetes",["Site A","Site B","Site C"])]

def add_direction_arrows(ax, better_side, y=-0.1, color="#C0392B",
                         fontsize=9, gap=0.012, max_len=0.2):
    xlo, xhi = ax.get_xlim()
    f0 = (0.0 - xlo) / (xhi - xlo)                 # axes-fraction of x=0
    f0 = min(max(f0, 0.0), 1.0)
    Ll = max(0.0, min(max_len, f0 - gap - 0.02))          # left arrow length
    Lr = max(0.0, min(max_len, (1 - f0) - gap - 0.02))    # right arrow length
    tr = ax.transAxes
    ap = dict(arrowstyle="-|>", color=color, lw=2)

    if Ll > 0.02:                                   # left-pointing arrow
        ax.annotate("", xy=(f0 - gap - Ll, y), xytext=(f0 - gap, y),
                    xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)
    if Lr > 0.02:                                   # right-pointing arrow
        ax.annotate("", xy=(f0 + gap + Lr, y), xytext=(f0 + gap, y),
                    xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)

    left_label  = "Shared better       " if better_side == "left"  else "Shared worse"
    right_label = "Shared worse"  if better_side == "left"  else "Shared better"
    if Ll > 0.02:
        ax.text(f0 - gap - Ll/2, y - 0.02, left_label, transform=tr,
                ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)
    if Lr > 0.02:
        ax.text(f0 + gap + Lr/2, y - 0.02, right_label, transform=tr,
                ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)

def build(metric, xlabel, xlim, note, fname, better_side):
    rows=[]; ypos=[]; ylab=[]; spans={}; y=0.0
    for t,ss in order:
        start=y
        for s in ss: rows.append((t,s)); ypos.append(y); ylab.append(s); y+=1
        spans[t]=(start,y-1); y+=0.8
    ymax=y
    ypos=[ymax-1-p for p in ypos]
    for t in list(spans): a,b=spans[t]; spans[t]=(ymax-1-b,ymax-1-a)
    fig,axes=plt.subplots(1,2,figsize=(12,7.4),sharey=True)
    off=0.16
    for ax,setting in zip(axes,["WN","LSO"]):
        for i,(t,s) in enumerate(rows):
            fl,ce=D[(t,s)][metric][setting]; yy=ypos[i]
            ax.errorbar(fl[0],yy+off,xerr=[[fl[0]-fl[1]],[fl[2]-fl[0]]],fmt='o',color=FLc,ms=5,capsize=2.5,lw=1.3,ecolor=FLc,zorder=3)
            ax.errorbar(ce[0],yy-off,xerr=[[ce[0]-ce[1]],[ce[2]-ce[0]]],fmt='s',color=CENTc,ms=5,capsize=2.5,lw=1.3,ecolor=CENTc,zorder=3)
        ax.axvline(0,color="#444",lw=1,ls=(0,(4,3)),zorder=1)
        ax.axvspan(-0.01,0.01,color="#EAF1F8",alpha=0.6,zorder=0)
        ax.set_xlim(*xlim); ax.set_ylim(-0.6,ymax-0.4)
        ax.set_yticks(ypos); ax.set_yticklabels(ylab,fontsize=9)
        ax.tick_params(axis='x',labelsize=9); ax.set_xlabel(xlabel,fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        for t,(lo,hi) in spans.items():
            ax.text(xlim[0]+0.012*(xlim[1]-xlim[0]),(lo+hi)/2,t,rotation=90,va='center',ha='center',fontsize=10.5,fontweight='bold',color="#333")
        add_direction_arrows(ax, better_side)  # <-- add arrows to each panel
    axes[0].set_title("Fidelity recovery  (site-contributing)",fontsize=12,fontweight='bold',pad=8)
    axes[1].set_title("Fidelity transportability  (site-withheld)",fontsize=12,fontweight='bold',pad=8)
    axes[0].annotate(note,xy=(0,ymax-0.5),xytext=(0.012*(xlim[1]-xlim[0]),ymax-0.15),fontsize=8,color="#444",va='top')
    leg=[Line2D([0],[0],marker='o',color='w',markerfacecolor=FLc,markersize=8,label='FL (HistAgg) \u2212 local'),
         Line2D([0],[0],marker='s',color='w',markerfacecolor=CENTc,markersize=8,label='Centralised \u2212 local')]
    fig.legend(handles=leg,loc='upper center',ncol=2,frameon=False,fontsize=10,bbox_to_anchor=(0.5,0.98))
    plt.tight_layout(rect=[0,0,1,0.945])
    plt.savefig(fname+".png",dpi=200,bbox_inches='tight'); plt.savefig(fname+".pdf",bbox_inches='tight'); plt.close()
    print("saved",fname)

build("AUC","AUC difference  (shared \u2212 local)",(-0.29,0.08),
      " ","fig3_auc_forest", better_side="right")
build("BRIER","Brier difference (shared \u2212 local)",(-0.06,0.30),
      " ","fig3_brier_forest", better_side="left")