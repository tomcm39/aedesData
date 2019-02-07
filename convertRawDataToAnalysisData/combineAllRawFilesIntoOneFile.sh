#mcandrew
cat ../rawData/* > ../rawData/allRawData.csv && echo 'data combined' &&
awk -F, '{ if(NR==1){flag=$2; print $0} else if($2!=flag) {print $0}}' ../rawData/allRawData.csv > ../rawData/allRawDataNoHeader.csv && echo 'remove headers' && 
gzip ../rawData/allRawDataNoHeader.csv && echo 'data zipped'
rm ../rawData/allRawData.csv
