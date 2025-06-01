import datetime
import logging

logmsg = ''
timemark = dict()
saveDefault = False

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('res.log')
file_handler.setLevel(logging.DEBUG)

def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	logger.debug(tem)

	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()

def SpentTime(marker, time):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker]

def SpentTooLong(marker, time, day=0, hour=0, minute=0, second=0):
	global timemark
	if marker not in timemark:
		msg = 'LOGGER ERROR, marker', marker, ' not found'
		tem = '%s: %s' % (time, msg)
		print(tem)
		return False
	return datetime.datetime.now() - timemark[marker] >= datetime.timedelta(days=day, hours=hour, minutes=minute, seconds=second)

if __name__ == '__main__':
	log('')