from pyspark.sql import SparkSession
from pyspark.sql.functions import (count, min, max, when, sqrt, sum, row_number, col,
                                   floor, sin, cos, atan2, radians, unix_timestamp
                                   )
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window


def q1(df):

    # Combining Date and Time columns into a single column
    df = df.withColumn('Data_Time', F.concat_ws(' ', df['Date'], df['Time']))

    # Converting string to timestamp
    df = df.withColumn('Data_Time', F.expr("to_timestamp(Data_Time, 'yyyy-MM-dd HH:mm:ss')"))

    # Adding 8 hours to the timestamp
    df = df.withColumn("Data_Time", F.expr("Data_Time + interval 8 hours"))

    # Extracting Date and Time from the timestamp
    df = df.withColumn('Date', F.col('Data_Time').cast('date'))
    df = df.withColumn('Time', F.expr("date_format(Data_Time, 'HH:mm:ss')"))

    # Calculating timestamp difference from the start date
    start_date = F.lit("1899-12-30 00:00:00")
    df = df.withColumn('Data_Time', unix_timestamp('Data_Time', 'yyyy-MM-dd HH:mm:ss'))
    df = df.withColumn('StartDate', unix_timestamp(start_date))
    df = df.withColumn('Timestamp', (F.col('Data_Time') - F.col('StartDate')) / F.lit(86400))

    # Dropping unnecessary columns
    df = df.drop('StartDate')
    df.drop('Data_Time').show()

    return df


def q2(df):
    # Count data points per day for each user
    data_points_per_day = df.groupBy('UserID', 'Date').count()

    # Filter to find days where a user has at least five data points
    at_least_five = data_points_per_day.filter(col('count') >= 5)

    # Count the number of days with at least five data points for each user
    days_with_at_least_five = at_least_five.groupBy('UserID').agg(count('*').alias('DaysCount'))

    # Sort users based on the number of days they have at least five data points
    top_users = days_with_at_least_five.sort(col('DaysCount').desc(), col('UserID').asc())

    # Display the top 5 users
    top_users.show(5)

    # Return the top 5 users
    return top_users.limit(5)


def q3(df):
    # Convert Timestamp to days and then to week numbers
    df = df.withColumn('Days', floor(col('Timestamp')))
    df = df.withColumn('Weeks', floor((col('Days') - 2) / 7))

    # Count data points per week for each user
    data_points_per_week = df.groupBy('UserID', 'Weeks').agg(count('*').alias('DataPointsCount'))

    # Filter out weeks where a user has more than 100 data points
    weeks_with_more_than_hundred = data_points_per_week.filter(col('DataPointsCount') > 100)

    # Count the number of such weeks for each user
    weeks_count = weeks_with_more_than_hundred.groupBy('UserID').agg(count('*').alias('WeeksCount'))

    # Order the results by UserID
    df = weeks_count.orderBy('UserID')

    # Display the results
    df.show()

    # Return the dataframe
    return df


def q4(df):
    # Compute the southernmost latitude for each user
    southernmost_latitudes = df.groupBy('UserID').agg(min('Latitude').alias('MinLatitude'))

    # Rename UserID column for joining purposes
    southernmost_latitudes = southernmost_latitudes.withColumnRenamed('UserID', 'ID')

    # Join the original dataframe with the southernmost latitudes to get the corresponding dates
    southernmost_points = df.join(
        southernmost_latitudes,
        (df.UserID == southernmost_latitudes.ID) & (df.Latitude == southernmost_latitudes.MinLatitude)
    )

    # Remove duplicate entries to ensure only one record day per user
    southernmost_points = southernmost_points.orderBy('UserID','Date').dropDuplicates(['UserID'])

    # Order by latitude to identify the southernmost points
    southernmost_points = southernmost_points.orderBy(col('MinLatitude'))

    # Select relevant columns and limit to first 5 entries
    df = southernmost_points.select('UserID', 'MinLatitude', 'Date').limit(5)

    # Display the results
    df.show()

    # Return the modified dataframe
    return df


def q5(df):
    # Convert altitude from feet to meters and filter out invalid values (-777)
    df = df.withColumn(
        'Altitude',
        when(col('Altitude') != F.lit(-777), col('Altitude') * 0.3048).otherwise(None)
    ).dropna()

    # Ensure Altitude is treated as a float
    df = df.withColumn('Altitude', col('Altitude').cast(DoubleType()))

    # Compute max and min altitude per day for each user
    altitude_stats_per_day = df.groupBy('UserID', 'Date').agg(
        max('Altitude').alias('MaxAltitude'),
        min('Altitude').alias('MinAltitude')
    )

    # Calculate altitude span (difference between max and min) per day
    altitude_span_per_day = altitude_stats_per_day.withColumn(
        'AltitudeSpan', col('MaxAltitude') - col('MinAltitude')
    )

    # Aggregate to find the maximum altitude span for each user
    df = altitude_span_per_day.groupBy('UserID').agg(max('AltitudeSpan').alias('AltitudeSpan'))

    # Order by the altitude span in descending order and limit to top 5
    df = df.orderBy(col('AltitudeSpan').desc()).limit(5)

    # Display the results
    df.show()

    # Return the dataframe
    return df


def dis_two_point(df):
    # Define window specification to order data by timestamp for each user and date
    windowSpec = Window.partitionBy('UserID', 'Date').orderBy('Timestamp')

    # Use the lag function to get the previous latitude and longitude for each point
    df = df.withColumn('PrevLatitude', F.lag('Latitude').over(windowSpec))
    df = df.withColumn('PrevLongitude', F.lag('Longitude').over(windowSpec))
    df = df.withColumn('PrevData_Time', F.lag('Data_Time').over(windowSpec))
    df = df.withColumn('dla', radians(col('Latitude') - col('PrevLatitude')))
    df = df.withColumn('dlo', radians(col('Longitude') - col('PrevLongitude')))

    # Calculate the distance
    df = df.withColumn('a',
                       sin(col('dla') / 2) ** 2 + cos(radians(col('PrevLatitude'))) * cos(radians(col('Latitude'))) * (
                           sin(col('dlo') / 2)) ** 2)
    df = df.withColumn('c', 2 * atan2(sqrt(col('a')), sqrt(1 - col('a'))))
    df = df.withColumn('Distance', 6373.0 * col('c')).drop('a', 'c', 'dla', 'dlo')

    # Drop rows with any null values and the previous latitude and longitude columns
    df = df.dropna(how='any').drop('PrevLatitude', 'PrevLongitude')

    # Return the updated dataframe
    return df


def q6(df):
    # Use the dis_two_point function to calculate distances between consecutive points
    daily_distance_df = dis_two_point(df)

    # Aggregate the distances to get the total daily distance for each user
    daily_distance_df = daily_distance_df.groupBy('UserID', 'Date').agg(sum('Distance').alias('DailyDistance'))

    # Define a window specification to rank users based on their maximum daily distance
    windowSpecUser = Window.partitionBy('UserID').orderBy(col('DailyDistance').desc(), 'Date')

    # Find the maximum daily distance for each user
    max_distance_by_user = daily_distance_df.withColumn('Rank', row_number().over(windowSpecUser)) \
        .filter(col('Rank') == 1).drop('Rank')

    # Display the date of maximum distance for each user
    max_distance_by_user.drop('DailyDistance').show()

    # Calculate the total distance traveled by all users
    total_distance = daily_distance_df.groupBy().agg(sum('DailyDistance').alias('TotalDistance'))
    # Display the total distance
    total_distance.show()

    # Return the maximum daily distance by user and the total distance
    return max_distance_by_user, total_distance


def q7(df):
    # Calculate the distance between consecutive points and the time difference in seconds
    df = dis_two_point(df.orderBy('UserID', 'Timestamp'))
    df = df.withColumn('Second', col('Data_Time') - col('PrevData_Time'))

    # Calculate the speed in km/h (distance in km divided by time in hours)
    df = df.withColumn('Speed', col('Distance') / col('Second') * 3600)

    # Aggregate to find the maximum speed for each user on each day
    df = df.groupBy('UserID', 'Date').agg(max('Speed').alias('MaxSpeed'))

    # Define a window specification to rank users based on their maximum speed
    windowSpecUser = Window.partitionBy('UserID').orderBy(col('MaxSpeed').desc(), 'Date')

    # Identify the maximum speed achieved by each user
    max_speed_by_user = df.withColumn('Rank', row_number().over(windowSpecUser)) \
        .filter(col('Rank') == 1).drop('Rank')

    # Display the maximum speed for each user
    max_speed_by_user.show()

    # Return the dataframe with the maximum speed for each user
    return max_speed_by_user


if __name__ == '__main__':
    spark = SparkSession.builder.appName("DataFrame").getOrCreate()
    df = spark.read.format('csv').option("header", "true").load('dataset.txt')

    df = q1(df)
    q2(df)
    q3(df)
    q4(df)
    q5(df)
    q6(df)
    q7(df)
