import ovrolwasolar.file_handler as fh
from ovrolwasolar.file_handler import File_Handler

time_duration='2023/05/02/18:00:00~2023/05/02/18:30:00'
freqstr=['78MHz','31MHz']
file_path='20230502_solar'
fp=File_Handler(time_duration=time_duration,freqstr=freqstr,file_path=file_path)

fp.start=fp.parse_duration()
fp.end=fp.parse_duration(get_end=True)


fp.get_selfcal_times_paths()



filename='a'

while filename is not None:
	filename=fp.get_current_file_for_selfcal('78MHz')
	print (filename)

filename='a'	
while filename is not None:
	filename=fp.get_current_file_for_imaging('78MHz')
	print (filename)
	
filename='a'

while filename is not None:
	filename=fp.get_current_file_for_selfcal('31MHz')
	print (filename)

filename='a'	
while filename is not None:
	filename=fp.get_current_file_for_imaging('31MHz')
	print (filename)

