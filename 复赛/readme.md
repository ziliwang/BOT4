### structure
 - raw_xxx: origin date
 - train_cal: store train process data

### solution
1. the current stock price effected by transaction of previous days. So I use right part(300x100 px) for training.
2. As first competition result, weekday is a strong feature to stock price predictions.
3. as my teammate sugested, 120 day price may be useful in stock price prediction.
4. stock price k-chart is a structured data presentation, there were less details in the picture, so I choosed a shallow network to this job, which constructed by a big convolution layer to catch the wide-range transaction variation and a mean pool to catch the global transaction. The output were  contacted  and inpu to a fully connected layer to catch the linear combination of the features. Last, the output of fully connected layer were inputed into another fully connected layer coupled with price features and weekday features.
5. the picture has different part, so I cat picture into three part: k-chart, Volume and trend.

![绘图1](/assets/绘图1.png)

### files description
 - xxx_low: low filter number
 - xxx_xl: extra low filter number
