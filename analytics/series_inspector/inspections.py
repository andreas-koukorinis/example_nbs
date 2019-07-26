class Inspection(object):
    EMAILS = []

    def __init__(self):
        pass

    def inspect_series(self, start_dt, end_dt):
        """
        Check one or more series between start_dt and end_dt (included) and return a status report which is sent by
        email

        :param start_dt: datetime
        :param end_dt: datetime
        :return: [InspectionResult] IT will send one email for each fo these
        """
        return []
