#!/bin/bash

# Check if the folder number is provided as a command line argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide the folder number as an argument."
    exit 1
fi

# Assign the first command line argument to the folder number
folder_number=$1

# Move everything from val to train folder to create the appropriate val folder
mv /home/dataset/val/images/* /home/dataset/train/images/;
mv /home/dataset/val/labels/* /home/dataset/train/labels/;

# Perform if statements based on the folder number
if [ $folder_number -eq 1 ]; then
    echo "Validation Fold $folder_number:"
    # Fog, Fog, Night, Night, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{012*,014*,009*,019*,001*,002*,003*,004*,005*,006*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{012*,014*,009*,019*,001*,002*,003*,004*,005*,006*} /home/dataset/val/labels/
elif [ $folder_number -eq 2 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Night, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{017*,020*,022*,034*,007*,008*,010*,011*,013*,015*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{017*,020*,022*,034*,007*,008*,010*,011*,013*,015*} /home/dataset/val/labels/
elif [ $folder_number -eq 3 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Night, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{021*,023*,043*,047*,016*,018*,025*,026*,027*,029*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{021*,023*,043*,047*,016*,018*,025*,026*,027*,029*} /home/dataset/val/labels/
elif [ $folder_number -eq 4 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Night, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{024*,028*,050*,053*,030*,031*,032*,033*,035*,036*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{024*,028*,050*,053*,030*,031*,032*,033*,035*,036*} /home/dataset/val/labels/
elif [ $folder_number -eq 5 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Night, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{040*,045*,068*,070*,037*,038*,039*,041*,042*,044*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{040*,045*,068*,070*,037*,038*,039*,041*,042*,044*} /home/dataset/val/labels/
elif [ $folder_number -eq 6 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Night, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{051*,055*,075*,095*,046*,048*,049*,052*,054*,056*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{051*,055*,075*,095*,046*,048*,049*,052*,054*,056*} /home/dataset/val/labels/
elif [ $folder_number -eq 7 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Night, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{061*,062*,085*,100*,058*,059*,060*,063*,064*,065*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{061*,062*,085*,100*,058*,059*,060*,063*,064*,065*} /home/dataset/val/labels/
elif [ $folder_number -eq 8 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Day, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{069*,072*,086*,067*,071*,073*,074*,076*,077*,078*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{069*,072*,086*,067*,071*,073*,074*,076*,077*,078*} /home/dataset/val/labels/
elif [ $folder_number -eq 9 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Day, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{081*,082*,087*,079*,080*,083*,084*,088*,089*,090*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{081*,082*,087*,079*,080*,083*,084*,088*,089*,090*} /home/dataset/val/labels/
elif [ $folder_number -eq 10 ]; then
    echo "Validation Fold  $folder_number:"
    # Fog, Fog, Night, Day, Day, Day, Day, Day, Day, Day
    mv /home/dataset/train/images/{093*,097*,092*,091*,094*,096*,098*,099*,057*,066*} /home/dataset/val/images/
    mv /home/dataset/train/labels/{093*,097*,092*,091*,094*,096*,098*,099*,057*,066*} /home/dataset/val/labels/
fi
