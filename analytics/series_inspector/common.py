
class InspectionStatus(object):
    # The inspection found no error
    OK = 'OK'
    # the inspection found some error
    ERROR = 'ERROR'
    # the inspection crashed
    CRASH = 'CRASH'


class InspectionResult(object):
    def __init__(self, status, brief_desc, full_desc):
        self.status = status
        self.brief_desc = brief_desc
        self.full_desc = full_desc


class RichInspectionResult(InspectionResult):
    def __init__(self, status, brief_desc, full_desc, inspection_class_name, inspection_kwargs, emails):
        super(RichInspectionResult, self).__init__(status, brief_desc, full_desc)
        self.inspection_class_name = inspection_class_name
        self.inspection_kwargs = inspection_kwargs
        self.emails = emails

    def __str__(self):
        return "<RichInspectionResult:{} status:{} brief_desc:{}>".format(
            self.inspection_class_name, self.status, self.brief_desc)
