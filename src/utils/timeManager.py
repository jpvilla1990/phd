from datetime import datetime, timedelta

class TimeManager(object):
    """
    Class to handle time
    """
    def getPeriodSeconds(periodicity : dict):
        """
        Get periodicity in seconds
        """
        periodicitySeconds : int = 0
        for key in periodicity:
            if key == "sec":
                periodicitySeconds += periodicity[key]
            elif key == "min":
                periodicitySeconds += periodicity[key] * 60
            elif key == "hour":
                periodicitySeconds += periodicity[key] * 60 * 60
            elif key == "day":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24
            elif key == "week":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24 * 7
            elif key == "month":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24 * (365 / 12)
            elif key == "year":
                periodicitySeconds += periodicity[key] * 60 * 60 * 24 * 365

        return periodicitySeconds

    def nextTimeStamp(timestamp : str, formatTime : str, period : int) -> str:
        """
        Method to add period to timestamp to generate next timestamp
        """
        return (datetime.strptime(timestamp, formatTime) + timedelta(seconds=period)).strftime(formatTime)

