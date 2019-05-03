import collections
import json
import pickle
import time
from datetime import datetime

import gpudb
import pandas
import pytz
from sklearn import compose
from sklearn import metrics
from sklearn import neural_network
from sklearn import preprocessing
from tzwhere import tzwhere


def predict(in_map, db=None, logger=None):
    _ensure_models_table(db, logger)
    model_id, model, timestamp = _get_model(db, logger)
    X, y = _get_training_data(db, timestamp)
    new_samples_cnt = len(X)
    logger.info('Num new samples: %d' % new_samples_cnt)

    if new_samples_cnt > 100:
        logger.info('Updating model')
        test_r2, train_r2 = _update_model(model, timestamp, X, y)
        logger.info('Test R2: %.5f' % test_r2)
        logger.info('Train R2: %.5f' % train_r2)
        logger.info('Saving model')
        _save_model(db, model_id + 1, model, test_r2, train_r2, new_samples_cnt)
        logger.info('Model saved')

    in_df = pandas.DataFrame([in_map])
    features_transformed = model[0].transform(in_df)
    predicted_count = model[1].predict(features_transformed)
    out_map = {'predicted_yes_responses': round(predicted_count[0])}
    return out_map


def _ensure_models_table(db, logger):
    """
    Make sure that there is a table for storing trained prediction models.
    The table is created if it doesn't exist.

    :param gpudb.GPUdb db: Kinetica DB connection
    """
    table_name = 'prediction_model'
    table_check = db.has_table(table_name=table_name)
    table_structure = [
        ['model_id', 'int', 'primary_key', 'shard_key'],
        ['dump', 'bytes'],
        ['created_on', 'long', 'timestamp'],
        ['sample_cnt', 'long'],
        ['test_r2', 'double'],
        ['train_r2', 'double']
    ]

    if not table_check['table_exists']:
        logger.info('Table %s for storing trained prediction models does not exist' % table_name)
        gpudb.GPUdbTable(_type=table_structure, name=table_name, db=db)
        logger.info('Table %s created' % table_name)
    else:
        logger.info('Table %s for storing trained prediction models already exists' % table_name)


def _get_model(db, logger):
    """
    Create prediction model.

    The model is defined as a two-step pipeline:
     - one-hot encoder for city, hour, day_of_week and country features,
     - and a simple neural network for regression.

    :param gpudb.GPUdb db: Kinetica DB connection
    :rtype: (int, pipeline.Pipeline, int)
    """

    model_records = db.get_records_and_decode(
        table_name='prediction_model', offset=0, limit=1,
        options={'sort_by': 'created_on', 'sort_order': 'descending'})

    if len(model_records['records']) > 0:
        logger.info('Model found in DB')
        model = model_records['records'][0]
        classifier = pickle.loads(model['dump'])
        return model['model_id'], classifier, model['created_on']
    else:
        logger.info('No model found in the DB, creating new one from scratch')
        column_transformer = compose.ColumnTransformer([
            ('oh', preprocessing.OneHotEncoder(handle_unknown='ignore'), ['city', 'hour', 'day_of_week', 'country']),
            ('do_nothing', preprocessing.MinMaxScaler(), ['group_members', 'group_events'])
        ])
        classifier = neural_network.MLPRegressor(hidden_layer_sizes=(1500, 750, 375), max_iter=1000, shuffle=True)
        return 0, (column_transformer, classifier), None


def _get_training_data(db, prev_model_timestamp):
    """
    Get data that will be used to updatimport pandase the prediction model.
    Only data collected from previous model update are collected, unless `prev_model_timestamp` is set to None

    :param gpudb.GPUdb db: Kinetica DB connection
    :param int prev_model_timestamp: Timestamp of previous model update. Set to None to get all data in the DB

    :returns: (X, y) tuple where X is a DataFrame containing training samples and y is an array of target values
    :rtype: (pandas.DataFrame, numpy.ndarray)
    """
    tzfinder = tzwhere.tzwhere()
    events = _get_events_from_db(db, prev_model_timestamp)

    if len(events['column_1']) == 0:
        return [], []

    events_df = _events_to_dataframe(events)
    events_df['timezone'] = events_df.apply(lambda e: tzfinder.tzNameAt(e['lat'], e['lon']), axis=1)
    events_df.dropna(inplace=True)
    events_df['timezone'] = events_df.apply(lambda e: pytz.timezone(e['timezone']), axis=1)
    events_df['time_local'] = events_df.apply(lambda e: e['time_utc'].astimezone(e['timezone']), axis=1)
    events_df['day_of_week'] = events_df.apply(lambda e: e['time_local'].weekday(), axis=1)
    events_df['hour'] = events_df.apply(lambda e: e['time_local'].hour, axis=1)

    y = events_df['yes_responses'].values
    X = events_df
    return X, y


def _update_model(model, timestamp, X, y):
    """
    :param (compose.ColumnTransformer, neural_network.MLPRegressor) model:
    :param int timestamp:
    :rtype: (float, float)
    """

    if timestamp is None:
        model[0].fit(X)
        test_score = 0.0
    else:
        y_pred = model[1].predict(model[0].transform(X))
        test_score = metrics.r2_score(y, y_pred)

    X = model[0].transform(X)
    model[1].partial_fit(X, y)
    y_pred_new = model[1].predict(X)
    train_score = metrics.r2_score(y, y_pred_new)

    return test_score, train_score


def _get_events_from_db(db, from_timestamp=None):
    """
    :param gpudb.GPUdb db: Connection to Kinetica's GPUdb
    :param int from_timestamp: Only return events newer than this timestamp
    :rtype dict
    """
    time_filter = '' if from_timestamp is None else ' AND event_timestamp >= %d' % from_timestamp
    events_response = db.aggregate_group_by(
        table_name='event_rsvp',
        column_names=[
            'SUM(response) AS yes_responses', 'MAX(city)', 'event_id', 'event_timestamp',
            'lat', 'lon', 'MAX(country)', 'MAX(group_members) AS group_members', 'MAX(group_events) AS group_events'],
        limit=100000,
        offset=0,
        encoding='json',
        options={
            'expression': 'IS_NULL(city) = 0 AND event_timestamp < NOW() '
                          'AND IS_NULL(country) = 0 AND IS_NULL(group_events) = 0 AND IS_NULL(group_members) = 0'
                          '%s' % time_filter,
            'having': 'SUM(response) >= 10'
        })
    events_response_dict = json.loads(events_response['json_encoded_response'])
    return events_response_dict


def _save_model(db, model_id, model, test_r2, train_r2, sample_cnt):
    model_record = collections.OrderedDict()
    model_record['model_id'] = model_id
    model_record['dump'] = pickle.dumps(model)
    model_record['created_on'] = int(time.time() * 1000)
    model_record['sample_cnt'] = sample_cnt
    model_record['test_r2'] = test_r2
    model_record['train_r2'] = train_r2

    table = gpudb.GPUdbTable(name='prediction_model', db=db)
    table.insert_records(model_record)


def _events_to_dataframe(events):
    """
    :param dict events:
    :rtype: pandas.DataFrame
    """
    event_records = []
    num_events = len(events['column_1'])

    for i in range(num_events):
        event_records.append({
            'city': events['column_2'][i],
            'event_id': events['column_3'][i],
            'time_utc': datetime.fromtimestamp(events['column_4'][i] / 1000, tz=pytz.utc),
            'yes_responses': events['column_1'][i],
            'lat': events['column_5'][i],
            'lon': events['column_6'][i],
            'country': events['column_7'][i].lower(),
            'group_members': events['column_8'][i],
            'group_events': events['column_9'][i],
        })

    df = pandas.DataFrame(event_records)
    return df
