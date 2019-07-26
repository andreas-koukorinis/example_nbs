import json
import time
import traceback
import logging
import multiprocessing as mp
from jinja2 import Template
from tabulate import tabulate
from sgmtradingcore.analytics.series_inspector.config import INSPECTION_TO_RUN, KNOWN_INSPECTIONS
from sgmtradingcore.analytics.series_inspector.common import InspectionStatus, RichInspectionResult
from sgmtradingcore.core.notifications import send_trading_system_email


class InspectionProcessInput(object):
    def __init__(self, inspection_class_name, inspection_kwargs, result_queue, start_dt, end_dt, inspector_class):
        """
        :param inspection_class_name:
        :param inspection_kwargs:
        :param result_queue: Queue of RichInspectionResult to be sent to the main process
        :param start_dt: run the comparison from this timestamp.
        :param end_dt: run the comparison up to this timestamp, included
        """
        self.inspection_class_name = inspection_class_name
        self.inspection_kwargs = inspection_kwargs
        self.result_queue = result_queue
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.inspector_class = inspector_class


def run_inspection(process_input):
    """
    Run an Inspection, and return RichInspectionResult through a shared queue
    :param process_input: InspectionProcessInput
    """
    inspection_class_name = process_input.inspection_class_name
    inspection_kwargs = process_input.inspection_kwargs
    result_queue = process_input.result_queue
    start_dt = process_input.start_dt
    end_dt = process_input.end_dt
    logging.info("Running {}".format(inspection_class_name))
    inspector_class = process_input.inspector_class
    inspection_class = inspector_class.get_inspection_class(inspection_class_name)
    emails = inspection_class.EMAILS

    try:
        inspection = inspection_class(**inspection_kwargs)
        inspection_results = inspection.inspect_series(start_dt, end_dt)
    except Exception as e:
        full_desc = traceback.format_exc()
        logging.error("Error on {}: \n{}".format(inspection_class_name, full_desc))

        inspection_results = [
            RichInspectionResult(
                InspectionStatus.CRASH,
                str(e),
                full_desc,
                inspection_class_name,
                inspection_kwargs,
                emails)
        ]

    if not isinstance(inspection_results, list):
        inspection_results = [inspection_results]

    for result in inspection_results:
        result_queue.put(RichInspectionResult(result.status, result.brief_desc, result.full_desc,
                                              inspection_class_name, inspection_kwargs, emails))
    return


class SeriesInspector(object):
    """
    Run each inspection in a different process, to avoid memory leak and connection limit
    """
    KNOWN_INSPECTIONS = KNOWN_INSPECTIONS

    def __init__(self, inspections_to_run=INSPECTION_TO_RUN):
        self.inspections_to_run = inspections_to_run

    @classmethod
    def get_inspection_class(cls, name):
        return [c for c in cls.KNOWN_INSPECTIONS if c.__name__ == name][0]

    def _check_valid_inspections(self):
        for inspection_class_name, inspection_kwargs in self.inspections_to_run:
            if inspection_class_name not in [c.__name__ for c in self.__class__.KNOWN_INSPECTIONS]:
                raise ValueError('Unknown Inspection {}'.format(inspection_class_name))

            try:
                json.dumps((inspection_class_name, inspection_kwargs))
            except:
                raise ValueError('Cannot serialize {} {}'.format(inspection_class_name, inspection_kwargs))
        for c in SeriesInspector.KNOWN_INSPECTIONS:
            if len(c.EMAILS) == 0:
                raise ValueError('Must specify some emails')

    def run(self, start_dt, end_dt):
        self._check_valid_inspections()

        result_queue = mp.Queue()
        self._run_all(result_queue, start_dt, end_dt)
        good_results, bad_results, crashed_results = self._get_results(result_queue)
        return good_results, bad_results, crashed_results

    @classmethod
    def send_emails(cls, good_results, bad_results, crashed_results, start_dt, end_dt):
        for bad_result in bad_results+crashed_results:
            send_bad_inspection_email(bad_result)

        if len(bad_results+crashed_results) > 0:
            send_inspection_recap_email(good_results+bad_results+crashed_results, start_dt, end_dt)

    def _run_all(self, result_queue, start_dt, end_dt):
        for inspection_class_name, inspection_kwargs in self.inspections_to_run:
            logging.info("About to run inspection {} for {}".format(inspection_class_name, inspection_kwargs))
            args = InspectionProcessInput(
                inspection_class_name, inspection_kwargs, result_queue, start_dt, end_dt, self.__class__)
            p = mp.Process(target=run_inspection, args=(args,))
            p.start()
            p.join()
            logging.info("Inspection finished")
        logging.info("All inspecttion run")

    @staticmethod
    def _get_results(result_queue):
        # inspection_results = [result_queue.get(block=True)]
        inspection_results = []
        while not result_queue.empty():
            inspection_results.append(result_queue.get(block=False))
            logging.info("Got result {}".format(inspection_results[-1]))

        good_results = [r for r in inspection_results if r.status == InspectionStatus.OK]
        bad_results = [r for r in inspection_results if r.status == InspectionStatus.ERROR]
        crashed_results = [r for r in inspection_results if r.status == InspectionStatus.CRASH]
        return good_results, bad_results, crashed_results


def send_inspection_recap_email(rich_results, start_dt, end_dt):
    """
    :param rich_results: [RichInspectionResult]
    :param start_dt: datetime
    :param end_dt: datetime
    """

    message = ""
    headers = ["status",
               "class",
               "brief",
               ]
    data = []
    for result in rich_results:
        line = ([result.status,
                 result.inspection_class_name,
                 result.brief_desc])

        data.append(line)
    message += tabulate(data, headers=headers, stralign='left')

    date_str = "From {} to {}   ({} minutes)".format(start_dt, end_dt, int((end_dt-start_dt).total_seconds()/60))

    template = Template("""
SeriesInspection report
{{ date_str }}
<pre>
{{ message }}
</pre>
        """)

    html_ = template.render(message=message, date_str=date_str)

    subject = 'SeriesInspection Report: {} to {}'.format(start_dt.date(), end_dt.date())
    to = ['lorenzo@stratagem.co', 'engineering@stratagem.co']
    _send_email_retry('SeriesInspection Report', html_, subject, to)


def send_bad_inspection_email(rich_inspection_result):
    """

    :param rich_inspection_result: RichInspectionResult
    """
    template = Template("""
Bad SeriesInspection {{ inspection_name }}
\nwith params:
{{ params }}

\nBrief:
{{ brief }}

\nMessage:
<pre>
{{ message }}
</pre>
        """)

    html_ = template.render(
        inspection_name=rich_inspection_result.inspection_class_name,
        params=str(rich_inspection_result.inspection_kwargs),
        brief=rich_inspection_result.brief_desc,
        message=rich_inspection_result.full_desc)

    subject = 'SeriesInspection {}: {}'.format(
        rich_inspection_result.status, rich_inspection_result.inspection_class_name)
    to = rich_inspection_result.emails

    _send_email_retry('Problem in SeriesInspection', html_, subject, to)


def _send_email_retry(text, html_, subject, to_, attachments=None):
    try:
        send_trading_system_email(text, html_, subject, to_, files=attachments)
    except:
        time.sleep(4)
        send_trading_system_email(text, html_, subject, to_, files=attachments)
