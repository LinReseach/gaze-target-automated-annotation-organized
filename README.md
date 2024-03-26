
This repo provide the implement code of our paper.
In our paper, we propose an annotation pipeline for automating this effort. In this work, we focus on videos in which the objects looked at do not move. As input for the proposed pipeline, we therefore only need to annotate object bounding boxes for the first frame of each video. The benefit, moreover, of manually annotating these frames is that we can also draw bounding boxes for objects outside of it, which enables estimating gaze targets in videos where not all objects are visible. A second issue that we address is that the models used for automating the pipeline annotate individual video frames. In practice, however, manual annotation is done at the event level for video segments instead of single frames. Therefore, we also introduce and investigate several variants of algorithms for aggregating frame-level to event-level annotations, which are used in the last step in our annotation pipeline.


input files
CVPR2020: pixel positions of bounding box, face pixel positions, images
l2cs: pixel positions of bounding, images


location of these input file
pixel positions of bounding box: data_fin

output file:
videos with lables on that, cvs(class)

--- file 


In bounding box
readbbtxt_invi (get dataframe from the txt document which include bounding box pixel position, for invisible data )
readbbtxt (get dataframe from the txt document which include bounding box pixel position, for visible data )
readbbtxt_lin (get dataframe from the txt document which include bounding box pixel position, for test data )
  
In L2CS
l2cs_final_version_vis+invis_omen_othersdata.py(process other data/test data in omen computer )
l2cs_final_version_vis+invis_omen.py (process visible and invisible data in omen computer)

In CVPR2020
baseline_lin_invi.py (deal with invisible data)
demolocal.py (deal with visible data. maybe need comment invisible, and uncommend visible command at the end part)
demolocal_lin.py(deal with other data.)
facedection_info_lin.py (input image and detect face box as output, for test data)
facedection_info.py(input image and detect face box as output, for vi and invi data)


readbbtxt,readbbtxtinv:convert pixel positions of bouding boxes(in txt) to dataframe


----run command
for cvpr2020: 
 install environment(github link)
 conda activate attention 
 python demo_local.py

for l2cs:
 install environment(github link)
 conda activate l2cs
  python3 demo_annotate_inv.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0 \
 --cam 0  




