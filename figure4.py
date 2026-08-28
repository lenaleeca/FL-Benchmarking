import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

# task, site, n (exact), FL - local Brier difference (within-network): diff, lo, hi
sites=[
 ("Sepsis","Site 1",6837, -0.002,-0.005,0.001),
 ("Sepsis","Site 2",2337,  0.000,-0.006,0.007),
 ("Sepsis","Site 3",3112, -0.005,-0.009,-0.001),
 ("Sepsis","Site 4", 993, -0.020,-0.035,-0.007),
 ("AMI","Site 1",4248,  0.001,-0.004,0.006),
 ("AMI","Site 2",1180, -0.006,-0.017,0.005),
 ("AMI","Site 3", 442, -0.004,-0.023,0.018),
 ("AMI","Site 4", 650,  0.000,-0.017,0.017),
 ("AMI","Site 5",1152,  0.004,-0.009,0.016),
 ("AMI","Site 6", 778, -0.007,-0.026,0.010),
 ("Diabetes","Site A",8018, -0.008,-0.011,-0.005),
 ("Diabetes","Site B",6228,  0.001,-0.002,0.004),
 ("Diabetes","Site C",4635,  0.000,-0.003,0.003),
]
style={"Sepsis":("#2C6FA6","o"),"AMI":("#B5651D","X"),"Diabetes":("#2E8B6B","D")}


def add_direction_arrows(ax, better_side, y=-0.1, color="#C0392B",
                         fontsize=9, gap=0.012, max_len=0.3):
 xlo, xhi = ax.get_xlim()
 f0 = (0.0 - xlo) / (xhi - xlo)  # axes-fraction of x=0
 f0 = min(max(f0, 0.0), 1.0)
 Ll = max(0.0, min(max_len, f0 - gap - 0.02))  # left arrow length
 Lr = max(0.0, min(max_len, (1 - f0) - gap - 0.02))  # right arrow length
 tr = ax.transAxes
 ap = dict(arrowstyle="-|>", color=color, lw=2)

 if Ll > 0.02:  # left-pointing arrow
  ax.annotate("", xy=(f0 - gap - Ll, y), xytext=(f0 - gap, y),
              xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)
 if Lr > 0.02:  # right-pointing arrow
  ax.annotate("", xy=(f0 + gap + Lr, y), xytext=(f0 + gap, y),
              xycoords=tr, textcoords=tr, arrowprops=ap, annotation_clip=False)

 left_label = "FL better" if better_side == "left" else "FL worse"
 right_label = "FL worse" if better_side == "left" else "FL better"
 if Ll > 0.02:
  ax.text(f0 - gap - Ll / 2, y - 0.02, left_label, transform=tr,
          ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)
 if Lr > 0.02:
  ax.text(f0 + gap + Lr / 2, y - 0.02, right_label, transform=tr,
          ha="center", va="top", color=color, fontsize=fontsize, clip_on=False)

fig,ax=plt.subplots(figsize=(6,7))
ax.set_yscale("log")                                             # sample size on y (log)
ax.axhspan(300,1500,color="#F4F4F0",alpha=0.7,zorder=0)          # small sites n<1500
ax.axvspan(-0.01,0.01,color="#EAF1F8",alpha=0.7,zorder=0)        # +/-0.01 negligible-difference band
ax.axvline(0,color="#333",ls=(0,(5,3)),lw=1,zorder=1)           # 0 = recovers local
for t,s,n,d,lo,hi in sites:
    c,m=style[t]
    ax.errorbar(d,n,xerr=[[d-lo],[hi-d]],fmt=m,color=c,mfc=c,mec=c,ms=8,capsize=3,lw=1.2,elinewidth=1.2,zorder=4)
add_direction_arrows(ax, "left")  # <-- add arrows to each panel
ax.set_xlim(-0.045,0.025); ax.set_ylim(360,9500)
ax.yaxis.set_major_locator(FixedLocator([500,1000,2000,4000,8000]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FixedFormatter(["500","1000","2000","4000","8000"]))
ax.set_xlabel("Brier difference  (FL \u2212 local)",fontsize=10)
ax.set_ylabel("Site sample size (n, log scale)",fontsize=10)
ax.tick_params(labelsize=9); ax.spines[['top','right']].set_visible(False)
#ax.text(-0.03,8800,"\u2190 Shared better",fontsize=10,color="#C0392B",style='oblique',ha='left',va='top')
#ax.text(0.03,8800,"Shared worse \u2192",fontsize=10,color="#C0392B",style='oblique',ha='right',va='top')
#ax.text(0.0015,9350,"0 = recovers local",fontsize=8.5,color="#444",style='italic',ha='left',va='top')
ax.text(-0.043,1400,"sites with n < 1,500",fontsize=9,color="#666",ha='left',va='center')
handles=[Line2D([0],[0],marker=style[t][1],color='w',markerfacecolor=style[t][0],markeredgecolor=style[t][0],markersize=8,label=t) for t in ["Sepsis","AMI","Diabetes"]]
ax.legend(handles=handles,loc='upper left',fontsize=9,frameon=True,title="Clinical task",title_fontsize=9)
plt.tight_layout()
plt.savefig("fig_samplesize_brier.pdf",bbox_inches='tight')
print("saved")