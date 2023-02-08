mkdir code-tmp
cp -r installed-packages.txt knn_nosol.ipynb  requirements-python2.txt requirements.txt softmax_nosol.ipynb utils code-tmp
mkdir code-tmp/nndl
cp nndl/knn_nosol.py code-tmp/nndl/knn.py
cp nndl/softmax_nosol.py code-tmp/nndl/softmax.py
cp nndl/__init__.py code-tmp/nndl/
pushd code-tmp
zip ../hw2-code.zip -r "." -x "**__pycache__**"
popd
rm -r code-tmp
