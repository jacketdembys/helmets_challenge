import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch


gt_file_path = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/gt.txt"
gt_df = np.array(pd.read_csv(gt_file_path, header=None))
#print(gt_df)
counts_dict = pd.Series(gt_df[:,6]).value_counts()  
counts_index = counts_dict.index.tolist()
counts_values = counts_dict.values.tolist()


column_index = 6
target_value = 8
index = np.where(gt_df[:, column_index] == target_value)[0]
line = gt_df[index,:]
print(line)

names = {
  0: ["motorbike"],
  1: ["DHelmet"],
  2: ["DNoHelmet"],
  3: ["P1Helmet"],
  4: ["P1NoHelmet"],
  5: ["P2Helmet"],
  6: ["P2NoHelmet"],
  7: ["P0Helmet"],
  8: ["P0NoHelmet"],
}

counts = list(zip(counts_index, counts_values))
counts.append((6, 0))
#print(counts)
counts = np.array(counts) #, dtype=np.float64)
print(counts)

for i in range(0,9):
    #print(i)
    #print(counts[i,0])
    num = counts[i,0]
    names[num-1].append(counts[i,1])

print(names)

df = pd.DataFrame(names)
df.index = ['classes', 'instances']
print(df.T)




fig, ax = plt.subplots()
fig.subplots_adjust(wspace=0)
ax.pie(counts[:,1], labels=counts[:,0], autopct='%1.1f%%')
plt.show(block=False)



"""
# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.subplots_adjust(wspace=0)

# pie chart parameters
overall_ratios = counts[:,1]
labels = counts[:,0]
explode = [0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1]
# rotate so that first wedge is split by the x-axis
angle = -180 * overall_ratios[-1]
wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                     labels=labels, explode=explode)

# bar chart parameters
#age_ratios = [.33, .54, .07, .06]
age_ratios = counts[4:,1]
#age_labels = ['Under 35', '35-49', '50-65', 'Over 65']
age_labels = counts[4:,0]
bottom = 1
width = .2

# Adding from the top matches the legend.
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,)
                 #alpha=0.1 + 0.25 * j)
    #ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')
    ax2.bar_label(bc, labels=[f"{height:.2%}"], label_type='center')

ax2.set_title('Age of approvers')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2
center, r = wedges[0].center, wedges[0].r
bar_height = sum(age_ratios)

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(4)
ax2.add_artist(con)

# draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(4)

plt.show()
"""




"""
# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
fig.subplots_adjust(wspace=0)

# pie chart parameters
overall_ratios = [.27, .56, .17]
labels = ['Approve', 'Disapprove', 'Undecided']
explode = [0.1, 0, 0]
# rotate so that first wedge is split by the x-axis
angle = -180 * overall_ratios[0]
wedges, *_ = ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                     labels=labels, explode=explode)

# bar chart parameters
age_ratios = [.33, .54, .07, .06]
age_labels = ['Under 35', '35-49', '50-65', 'Over 65']
bottom = 1
width = .2

# Adding from the top matches the legend.
for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color='C0', label=label,
                 alpha=0.1 + 0.25 * j)
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

ax2.set_title('Age of approvers')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(- 2.5 * width, 2.5 * width)

# use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2
center, r = wedges[0].center, wedges[0].r
bar_height = sum(age_ratios)

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(4)
ax2.add_artist(con)

# draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(4)

plt.show()
"""


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    gt_file_path = "/home/retina/dembysj/Dropbox/WCCI2024/challenges/aicity2024_track5/aicity2024_track5_train/gt.txt"
    gt_df = np.array(pd.read_csv(gt_file_path, header=None))
    counts_dict = pd.Series(gt_df[:,6]).value_counts()    
    counts_index = counts_dict.index.tolist()
    counts_values = counts_dict.values.tolist()
    counts = list(zip(counts_index, counts_values))
    counts.append((6, 0))
    counts = np.array(counts)
    #counts = [[i, counts_dict[i]] for i in counts_dict[:]]

    sum_counts = counts[:,1].sum()
    #counts_percent = counts.copy()
    #counts_percent[:,1] = (counts_percent[:,1]/sum_counts)*100

    fig, ax = plt.subplots()
    ax.pie(counts[:,1], labels=counts[:,0], autopct='%1.1f%%')
    plt.show()

    
    #plt.xlabel("Classes")
    #plt.ylabel("Counts")
    #plt.title("Class distribution")
    #plt.show()
    #print(gt_df)
"""
