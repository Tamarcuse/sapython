FROM python:3.11.8
COPY . .
RUN pip install yahoofinancials \
    && pip install pandas \
    && pip install statsmodels \
    && pip install matplotlib
CMD ["python", "capm.py"]
