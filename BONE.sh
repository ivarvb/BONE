#!/bin/bash
ev="./python/python38bone/"
pathapp="./sourcecode/src/vx/bone/"
install () {
    #sudo apt-get install python3-venv
    #sudo apt install python3-pip
    #sudo apt-get install libsuitesparse-dev
    #sudo apt install libx11-dev
    #############sudo apt install nvidia-cuda-toolkit


    rm -r $ev
    mkdir $ev
    python3 -m venv $ev
    source $ev"bin/activate"

    # install packages
    # pip3 install -r requirements.txt
    pip3 install wheel
    pip3 install numpy
    pip3 install scikit-sparse
    pip3 install matplotlib
    pip3 install pandas
    pip3 install opencv-python
    pip3 install scikit-image
    pip3 install -U scikit-learn
    pip3 install xgboost
    pip3 install ujson
    pip3 install seaborn
    pip3 install cython
    pip3 install xgboost

    pip3 install SimpleITK
    pip3 install pyradiomics
    pip3 install thundersvm-cpu
    pip3 install sklearn-genetic
    pip3 install mlxtend
    pip3 install imblearn
    pip3 install tune_sklearn

}
#compile () {
#    source $ev"bin/activate"
#    cd ./sourcecode/src/vx/com/px/image/
#    sh Makefile.sh
#    
#}
#execute () {
#    source $ev"bin/activate"
#    cd ./sourcecode/src/vx/lha/
#    python3 Main.py
#}

masks () {
    source $ev"bin/activate"
    cd $pathapp
    python3 Mask.py
}
split () {
    source $ev"bin/activate"
    cd $pathapp
    python3 SplitDataset.py
}
extraction () {
    source $ev"bin/activate"
    cd $pathapp
    python3 FeatureExtraction.py
}
classification () {
    source $ev"bin/activate"
    cd $pathapp
    python3 Classification.py
}
ex () {
    source $ev"bin/activate"
    cd $pathapp
    python3 Experiments.py
}
gd () {
    source $ev"bin/activate"
    cd $pathapp
    python3 GDataset.py
}

args=("$@")
T1=${args[0]}
FILEINPUT=${args[1]}
if [ "$T1" = "install" ]; then
    install
elif [ "$T1" = "execute" ]; then
    execute
elif [ "$T1" = "masks" ]; then
    masks
elif [ "$T1" = "split" ]; then
    split
elif [ "$T1" = "extraction" ]; then
    extraction
elif [ "$T1" = "classification" ]; then
    classification
elif [ "$T1" = "ex" ]; then
    ex
elif [ "$T1" = "gd" ]; then
    gd
fi
