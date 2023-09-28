mkdir HMDB51
cd HMDB51
curl https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar -o hmdb51_org.rar
unrar x hmdb51_org.rar
cd ..

mkdir HMDB-AD
cd HMDB-AD
mkdir training testing training/videos testing/videos training/frames testing/frames

cd training/videos
unrar x ../../../HMDB51/walk.rar
unrar x ../../../HMDB51/run.rar

cd ../../testing/videos
unrar x ../../../HMDB51/climb.rar
unrar x ../../../HMDB51/cartwheel.rar

cd ../../../

mkdir HMDB-Violence
cd HMDB-Violence
mkdir training testing training/videos testing/videos training/frames testing/frames

cd training/videos
unrar x ../../../HMDB51/cartwheel.rar
unrar x ../../../HMDB51/walk.rar
unrar x ../../../HMDB51/climb.rar
unrar x ../../../HMDB51/run.rar
unrar x ../../../HMDB51/turn.rar
unrar x ../../../HMDB51/throw.rar
unrar x ../../../HMDB51/hug.rar
unrar x ../../../HMDB51/wave.rar
unrar x ../../../HMDB51/sit.rar

cd ../../testing/videos
unrar x ../../../HMDB51/fencing.rar
unrar x ../../../HMDB51/fall.rar
unrar x ../../../HMDB51/kick.rar
unrar x ../../../HMDB51/punch.rar
unrar x ../../../HMDB51/sword.rar
unrar x ../../../HMDB51/hit.rar
unrar x ../../../HMDB51/shoot.rar

cd ../../../

python configure_hmdb_datasets.py
python extract_frames.py HMDB-AD
python extract_frames.py HMDB-Violence