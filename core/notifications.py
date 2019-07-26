import smtplib
import threading
import warnings
from email import Encoders
from email.mime.multipart import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from threading import Thread

from datadog.api.constants import CheckStatus
from jinja2 import Template
from stratagemdataprocessing.apihooks.datadoghq import DataDogPusher

from sgmtradingcore.core.ts_log import get_logger
from sgmtradingcore.strategies.config.configurations import TRADING_USER_MAP

TRADING_SYSTEM_EMAIL = 'trading_system@stratagem.co'


def send_backtest_failure_email(strategy_name, strategy_desc, message, notify):
    template = Template("""
Error report for backtest run {{ run_name }}

<pre>
Error message was {{ message }}
</pre>
        """)
    html_ = template.render(
        run_name=strategy_desc, strategy=strategy_name, message=message)

    subject = 'Backtest run %s failed' % strategy_desc

    send_trading_system_email('Check your driver program logs for details', html_, subject, notify)


def send_backtest_success_email(strategy_name, strategy_desc, notify):
    template = Template("""
Success report for backtest run {{ run_name }}
        """)
    html_ = template.render(
        run_name=strategy_desc, strategy=strategy_name)

    subject = 'Backtest run %s finished succesfuly' % strategy_desc

    send_trading_system_email('All good', html_, subject, notify)


def send_realtime_failure_email(trading_user_id, strategy, strategy_run_ids, message, notify):
    template = Template("""
Crash notification for trading user {{ trading_user }} from strategy with descr {{ strategy_run_ids }}

<pre>
Error message was {{ message }}
</pre>
        """)

    trading_user_name = TRADING_USER_MAP.get(trading_user_id, 'unknown')
    strategy_run_ids_str = ', '.join(strategy_run_ids)
    html_ = template.render(
        trading_user=trading_user_name, strategy_run_ids=strategy_run_ids_str, message=message)

    subject = 'Crash notification for trading user %s from strategy %s, with descr %s' % (
        trading_user_name, strategy, strategy_run_ids_str)

    send_trading_system_email('Check the strategy logs for details', html_, subject, notify)


def send_realtime_exit_email(trading_user_id, strategy_run_ids, notify):
    template = Template("""
Exit notification for trading user {{ trading_user }} from strategy with descr {{ strategy_run_ids }}

<pre>
The strategy has exited normally and might need to be restarted
</pre>
        """)

    trading_user_name = TRADING_USER_MAP.get(trading_user_id, 'unknown')
    strategy_run_ids_str = ', '.join(strategy_run_ids)
    html_ = template.render(
        trading_user=trading_user_name, strategy_run_ids=strategy_run_ids_str)

    subject = 'Exit notification for trading user %s from strategy with descr %s' % (
        trading_user_name, strategy_run_ids_str)

    send_trading_system_email('Check the strategy logs for details', html_, subject, notify)


def send_algo_manager_email(trading_user_id, strategy_run_id, message, notify):
    template = Template("""
Algo manager notification for trading user {{ trading_user }} from strategy with descr {{ strategy_run_ids }}

<pre>
{{ message }}
</pre>
        """)

    trading_user_name = TRADING_USER_MAP.get(trading_user_id, 'unknown')
    strategy_run_ids_str = str(strategy_run_id)
    html_ = template.render(
        trading_user=trading_user_name, strategy_run_ids=strategy_run_ids_str, message=message)

    subject = 'Algo manager notification for trading user %s from strategy with descr %s' % (
        trading_user_name, strategy_run_ids_str)

    send_trading_system_email('Check the strategy logs for details', html_, subject, notify)


def send_trading_system_email(text, html_, subject, to_, files=None, multipart_type='alternative'):
    _send_email_generic(text, html_, subject, to_, TRADING_SYSTEM_EMAIL,
                        password='!5tr!t!93m!', files=files, multipart_type=multipart_type)


def _send_email_generic(text, html_, subject, to_, from_, password=None, files=None, multipart_type='alternative'):
    """
    Function to send email from a specific account.
    Use local mta agent if from_==TRADING_SYSTEM_EMAIL, otherwise needs login and password.
    A wrapper around this function such as send_trading_system_email is highly recommend to avoid having the password
    everywhere.

    :param text: (str) Text to include in the email content (added as <p>text</p> if html_ specified)
    :param html_: (str) Html to include in the email content (after text if specific)
    :param subject: (str) Subject of the email
    :param to_: (iterable of str) or (str)  Recipients of the emails
    :param from_: email address of the account to use to send email
    :param password: password of the account to use to send email
    :param files: (list of str) or (str) files paths + names to attach to the email
    :return: result of the send
    """

    if files is None:
        files = []
    if isinstance(files, str):
        files = [files]

    if html_ is not None and text is not None:
        html_ = '<p>' + text + '</p>' + html_
        text = None

    msg = MIMEMultipart(multipart_type)
    msg['Subject'] = subject
    msg['From'] = from_
    msg['To'] = to_ if isinstance(to_, basestring) else '; '.join(to_)

    if text is not None:
        msg_text = MIMEText(text, 'plain')
        msg.attach(msg_text)
    if html_ is not None:
        msg_html = MIMEText(html_, 'html', _charset='utf-8')
        msg.attach(msg_html)

    for f in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(f, "rb").read())
        Encoders.encode_base64(part)

        attachment_name = f.split('/')[-1]
        part.add_header('Content-Disposition', 'attachment; filename="' + attachment_name + '"')
        msg.attach(part)

    sent = False
    if TRADING_SYSTEM_EMAIL in from_:
        try:
            smtp = smtplib.SMTP('localhost')
            res_send = smtp.sendmail(from_, to_, msg.as_string())
            sent = True
        except Exception:
            pass

    if not sent:
        smtp = smtplib.SMTP('smtp.gmail.com', 587)
        smtp.ehlo()
        smtp.starttls()
        smtp.login(from_, password)
        res_send = smtp.sendmail(from_, to_, msg.as_string())

    smtp.quit()
    return res_send


def _send_email(text, html_, subject, to_, attachment=None, attachments=None, multipart_type='alternative'):
    """
    Send an arbitrary email.
    :param text:
    :param html_:
    :param subject:
    :param to_: string or list
    :param attachment: attachment filename/path; (opened and read below)
    :return:
    """
    warnings.warn('_send_email should not be used. Use send_trading_system_email instead.'
                  'Note that params attachement and attachements are merged into files param'
                  'This function will be deleted on the 2018-02-01',
                  DeprecationWarning)
    if attachments is None:
        attachments = []
    if attachment:
        attachments.append(attachment)

    send_trading_system_email(text, html_, subject, to_, files=attachments,
                              multipart_type=multipart_type)


class DataDogHeartBeat(object):
    def __init__(self, user_id, name=None, desc=None, code=None, interval=30.):
        self._user_id = user_id
        self._user_name = TRADING_USER_MAP[self._user_id]
        self._name = name
        self._desc = desc
        self._code = code
        self._datadog = None
        self._datadog_thread = None
        self._stop_flag = threading.Event()
        self._interval = interval
        self._logger = get_logger(self.__class__.__name__)

    def tag(self, status='up'):
        return 'account.{}.strategy.{}.{}.{}.{}'.format(self._user_name, self._name, self._desc, self._code, status)

    def start(self):
        self._datadog = DataDogPusher()
        tag = self.tag()

        def update_datadog():
            while True:
                self._logger.debug(None, 'Sending datadog heatbeat {}'.format(tag))
                self._datadog.service_check(tag, message='Heartbeat')
                while not self._stop_flag.wait(timeout=self._interval):
                    break

                if self._stop_flag.is_set():
                    return

        self._logger.info(None, "About to initialise datadog thread for tag {}.".format(tag))
        self._datadog_thread = Thread(target=update_datadog, name='Datadog Heartbeat Thread')
        self._datadog_thread.daemon = True
        self._datadog_thread.start()

    def stop(self, status='stopped'):
        if self._datadog_thread is not None:
            self._stop_flag.set()
            tag = self.tag()
            self._logger.info(None, "Sending end event for tag {}".format(tag))
            self._datadog_thread.join()
            self._datadog_thread = None
            self._datadog.service_check(tag, status=CheckStatus.WARNING, message=status)
