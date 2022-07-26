from datetime import datetime
from dateutil import parser
from pytz import utc
from pytz import timezone

eastern = timezone('US/Eastern')
created_at = '2021-01-21T23:55:29.000Z'
datetime_string = created_at.split(".")[0]
datetime_object = datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S')
aware_datetime = datetime_object.replace(tzinfo=utc)
eastern_time = aware_datetime.astimezone(eastern)
print(eastern_time)