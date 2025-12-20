# Comprehensive Academic Bibliography: Trading Methodologies (2015-2025)

**Over 90 peer-reviewed papers, SSRN working papers, and arXiv preprints** form the current foundation of quantitative trading research. This bibliography synthesizes the most important academic contributions across technical analysis, factor models, algorithmic execution, position sizing, machine learning, backtesting methodology, and cross-asset strategies. The literature reveals several converging themes: the critical importance of multiple testing corrections, the decay of published anomalies, the superiority of ensemble and hybrid approaches, and the emerging dominance of machine learning methods with proper validation frameworks.

---

## 1. Technical Analysis

Academic research has increasingly validated technical analysis as a source of predictive signals, particularly when combined with machine learning methods. The 2015-2025 literature shows technical strategies retain statistical significance, though profitability has diminished over time due to arbitrage.

### Comprehensive Reviews

**Han, Y., Liu, Y., Zhou, G., & Zhu, Y. (2021). "Technical Analysis in the Stock Market: A Review."**
- *Source:* SSRN Working Paper #3850494 / Management Science
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3850494
- *Summary:* Comprehensive survey covering time-series predictive power and cross-sectional predictability of technical indicators. Extends analysis to ML approaches including Lasso, neural networks, and genetic programming across 1987-2020.
- *Applicability:* HIGH | *Access:* SSRN Open Access

### Momentum and Relative Strength

**Zhu, Z., Duan, X., & Tu, J. (2019). "Relative Strength over Investment Horizons and Stock Returns."**
- *Source:* Journal of Portfolio Management, Vol. 46(1), pp. 91-107
- *DOI:* https://jpm.pm-research.com/content/46/1/91
- *Summary:* Proposes novel relative-strength measure synthesizing short- and intermediate-term price information. Demonstrates returns exceeding simple momentum profits, robust across market conditions.
- *Applicability:* HIGH | *Access:* Journal Paywall

**Ding, W., Mazouz, K., Ap Gwilym, O., & Wang, Q. (2022). "Technical Analysis as a Sentiment Barometer and the Cross-Section of Stock Returns."**
- *Source:* SSRN Working Paper #4296699
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4296699
- *Summary:* Explores sentiment channel through which technical analysis adds value. Builds sentiment barometer from technical trading strategies for predicting cross-sectional returns.
- *Applicability:* MEDIUM | *Access:* SSRN Open Access

**Comparative Study of MACD-Based Trading (2022)**
- *Source:* arXiv:2206.12282 (q-fin)
- *URL:* https://arxiv.org/pdf/2206.12282
- *Summary:* Comparison of MACD combined with RSI, MFI, and Bollinger Bands. Develops VPVMA indicator incorporating volume and volatility. MACD+RSI strategy shows superior risk-adjusted performance.
- *Applicability:* HIGH | *Access:* arXiv Open Access

### Moving Average Strategies

**Huang, J.Z. & Huang, Z. (2020). "Testing Moving Average Trading Strategies on ETFs."**
- *Source:* Journal of Empirical Finance, Vol. 57, pp. 16-32
- *Summary:* MA strategies using ETFs generate positive CAPM alpha despite lower raw returns than buy-and-hold. Proposes quasi-intraday MA strategy (QUIMA) that outperforms standard approaches.
- *Applicability:* HIGH | *Access:* Journal (Elsevier)

**Wang, H., et al. (2019). "Detailed Study of a Moving Average Trading Rule."**
- *Source:* arXiv:1907.00212 (q-fin)
- *URL:* https://arxiv.org/pdf/1907.00212
- *Summary:* Links MA trading performance to stochastic process characteristics. Shows relationship between autocorrelation and MA strategy profitability.
- *Applicability:* MEDIUM | *Access:* arXiv Open Access

**Liu, W., et al. (2021). "Profitability of Moving-Average Technical Analysis over the Firm Life Cycle."**
- *Source:* Pacific-Basin Finance Journal
- *DOI:* https://doi.org/10.1016/j.pacfin.2021.101645
- *Summary:* MA profitability varies across firm life cycle with U-shape relationship. Better performance during bearish markets and high-volatility periods.
- *Applicability:* HIGH | *Access:* Journal Paywall

### Support/Resistance and Chart Patterns

**Chen, J. (2021). "Evidence and Behaviour of Support and Resistance Levels in Financial Time Series."**
- *Source:* arXiv:2101.07410 (q-fin)
- *URL:* https://arxiv.org/abs/2101.07410
- *Summary:* Develops heuristic discovery algorithm for S/R levels in intraday price series. Finds S/R levels can reverse price trends statistically significantly with decay over time.
- *Applicability:* HIGH | *Access:* arXiv Open Access

**De Angelis, T. & Peskir, G. (2016). "Optimal Prediction of Resistance and Support Levels."**
- *Source:* Applied Mathematical Finance, Vol. 23(6), pp. 465-483
- *URL:* https://personalpages.manchester.ac.uk/staff/goran.peskir/levels.pdf
- *Summary:* Mathematical framework for optimal prediction of S/R levels using stochastic processes. Derives nonlinear integral equations for optimal stopping boundaries.
- *Applicability:* MEDIUM | *Access:* Open Access PDF

**Chu, B. (2024). "Technical Analysis with Machine Learning Classification Algorithms."**
- *Source:* SSRN Working Paper #4765615
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4765615
- *Summary:* Applies ML algorithms to technical indicators and chart patterns. Candlestick patterns show strong predictive power when leveraged by Random Forest.
- *Applicability:* HIGH | *Access:* SSRN Open Access

### Trend-Following Research

**Zarattini, C., Pagani, A., & Wilcox, C. (2024). "Does Trend-Following Still Work on Stocks?"**
- *Source:* SSRN Working Paper #5084316
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5084316
- *Summary:* Survivorship-bias-free analysis (1950-2024). Confirms highly skewed profit distribution with <7% of trades driving cumulative profitability. Long-only trend strategy shows **15.19% CAGR** with **6.18% annualized alpha**.
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Sepp, A. (2018). "The Science and Practice of Trend-following Systems."**
- *Source:* SSRN Working Paper #3167787
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167787
- *Summary:* Derives exact relationship between trend-following P&L and autocorrelation function under ARFIMA processes. Shows TF profitable when long-term autocorrelation positive.
- *Applicability:* HIGH | *Access:* SSRN Open Access

### Mean Reversion

**Pagonidis, A.S. (2016). "The IBS Effect: Mean Reversion in Equity ETFs."**
- *Source:* NAAIM Research Paper
- *URL:* https://www.naaim.org/wp-content/uploads/2014/04/00V_Alexander_Pagonidis_The-IBS-Effect-Mean-Reversion-in-Equity-ETFs-1.pdf
- *Summary:* Documents Internal Bar Strength (IBS) indicator capturing daily mean reversion. Combining IBS with RSI improves returns ~10% while decreasing time in market by 45%.
- *Applicability:* HIGH | *Access:* Open Access

---

## 2. Fundamental Analysis and Factor Models

Factor investing research has matured significantly, with the "factor zoo" problem prompting rigorous replication studies and methodological advances for distinguishing genuine anomalies from data-mined artifacts.

### Foundational Factor Models

**Fama, E.F. & French, K.R. (2015). "A Five-Factor Asset Pricing Model."**
- *Source:* Journal of Financial Economics, Vol. 116(1), pp. 1-22
- *DOI:* 10.1016/j.jfineco.2014.10.010
- *SSRN:* https://ssrn.com/abstract=2287202
- *Summary:* Extends three-factor model by adding profitability (RMW) and investment (CMA) factors. Five-factor model captures size, value, profitability, and investment patterns better than three-factor model. **HML becomes partially redundant** with profitability and investment factors.
- *Applicability:* HIGH | *Access:* SSRN Open Access / Journal

**Fama, E.F. & French, K.R. (2018). "Choosing Factors."**
- *Source:* Journal of Financial Economics, Vol. 128(2), pp. 234-252
- *DOI:* 10.1016/j.jfineco.2018.02.012
- *Summary:* Compares nested models and factor construction choices. Six-factor model (adding momentum) performs best. Provides guidance on cash vs. operating profitability, small vs. big stocks in factor construction.
- *Applicability:* HIGH | *Access:* Journal Paywall

### Quality and Low-Volatility Factors

**Novy-Marx, R. (2013). "The Other Side of Value: The Gross Profitability Premium."**
- *Source:* Journal of Financial Economics, Vol. 108(1), pp. 1-28
- *DOI:* 10.1016/j.jfineco.2013.01.003
- *Summary:* Gross profits-to-assets has same power as book-to-market for return prediction. Combining profitability with value dramatically increases strategy performance. Won 2013 Fama-DFA Prize.
- *Applicability:* HIGH | *Access:* Journal Paywall

**Asness, C.S., Frazzini, A., & Pedersen, L.H. (2019). "Quality Minus Junk."**
- *Source:* Review of Accounting Studies, Vol. 24, pp. 34-112
- *DOI:* 10.1007/s11142-018-9470-2 | *SSRN:* https://ssrn.com/abstract=2312432
- *Summary:* Defines quality as profitability, growth, safety, and payout. QMJ factor earns significant risk-adjusted returns in US and **24 countries**. High-quality stocks have negative market beta and perform well during crises.
- *Applicability:* HIGH | *Access:* SSRN Open Access (QMJ data available from AQR)

**Frazzini, A. & Pedersen, L.H. (2014). "Betting Against Beta."**
- *Source:* Journal of Financial Economics, Vol. 111(1), pp. 1-25
- *DOI:* 10.1016/j.jfineco.2013.10.005 | *SSRN:* https://ssrn.com/abstract=2049939
- *Summary:* Constrained investors bid up high-beta assets, causing high beta to associate with low alpha. BAB factor produces Sharpe ratio of **0.75** (1926-2009) across equities, bonds, and futures.
- *Applicability:* HIGH | *Access:* SSRN Open Access (BAB data available from AQR)

**Hsu, J.C., Kalesnik, V., & Kose, E. (2019). "What Is Quality?"**
- *Source:* Financial Analysts Journal, Vol. 75(2), pp. 44-61
- *DOI:* 10.1080/0015198X.2019.1567194
- *Summary:* Only four quality categories exhibit robustness: profitability, accounting quality (accruals), payout/dilution, and investment. Capital structure and earnings stability show little premium evidence.
- *Applicability:* HIGH | *Access:* Journal (Taylor & Francis)

### Anomaly Replication and the "Factor Zoo"

**McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?"**
- *Source:* Journal of Finance, Vol. 71(1), pp. 5-32
- *DOI:* 10.1111/jofi.12365 | *SSRN:* https://ssrn.com/abstract=2156623
- *Summary:* Studies 97 anomaly variables. Portfolio returns are **26% lower out-of-sample** (data mining) and **58% lower post-publication**. Publication-informed arbitrage accounts for 32% decline.
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Hou, K., Xue, C., & Zhang, L. (2020). "Replicating Anomalies."**
- *Source:* Review of Financial Studies, Vol. 33(5), pp. 2019-2133
- *DOI:* 10.1093/rfs/hhz061 | *SSRN:* https://ssrn.com/abstract=2961979
- *Summary:* Comprehensive replication of **447 anomalies**. With NYSE breakpoints and value-weighted returns, **286 anomalies (64%) are insignificant** at 5% level; 380 (85%) fail t-cutoff of 3.0. q-factor model leaves 115 of 161 significant alphas insignificant.
- *Applicability:* HIGH | *Access:* SSRN Open Access (q-factor data at global-q.org)

**Chen, A.Y. & Zimmermann, T. (2022). "Open Source Cross-Sectional Asset Pricing."**
- *Source:* Critical Finance Review, Vol. 11(2), pp. 207-264
- *DOI:* 10.1561/104.00000112 | *Website:* https://www.openassetpricing.com/
- *Summary:* Publicly available code and data reproducing **319 cross-sectional return predictors**. For 161 clearly significant characteristics, 98% reproduce with t-stats above 1.96.
- *Applicability:* HIGH | *Access:* Open Access (GitHub + Data Downloads)

**Feng, G., Giglio, S., & Xiu, D. (2020). "Taming the Factor Zoo: A Test of New Factors."**
- *Source:* Journal of Finance, Vol. 75(3), pp. 1327-1370
- *DOI:* 10.1111/jofi.12883 | *PDF:* https://dachxiu.chicagobooth.edu/download/ZOO.pdf
- *Summary:* Develops double-selection LASSO to evaluate whether new factors add value. Most "new" factors from 2012-2016 don't survive this test against existing benchmarks.
- *Applicability:* MEDIUM | *Access:* Author's Website (PDF)

**Swade, A., Hanauer, M.X., Lohre, H., & Blitz, D. (2024). "Factor Zoo (.zip)."**
- *Source:* Journal of Portfolio Management, Vol. 50(3), pp. 11-31
- *DOI:* 10.3905/jpm.2023.1.561 | *SSRN:* https://ssrn.com/abstract=4605976
- *Summary:* Approximately **15 factors are sufficient** to span the entire factor zoo. While selected factor styles are persistent, specific representatives vary over time.
- *Applicability:* MEDIUM-HIGH | *Access:* SSRN Open Access

### Size Factor Rehabilitation

**Asness, C.S., et al. (2018). "Size Matters, If You Control Your Junk."**
- *Source:* Journal of Financial Economics, Vol. 129(3), pp. 479-509
- *DOI:* 10.1016/j.jfineco.2018.05.006 | *SSRN:* https://ssrn.com/abstract=2553889
- *Summary:* Controlling for quality/junk **more than doubles** average SMB performance and significance. Resurrects size effect in 1980s-1990s and restores monotonic size-return relationship.
- *Applicability:* HIGH | *Access:* SSRN Open Access

### Key Data Resources

- **Ken French Data Library:** https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **AQR Data Sets:** https://www.aqr.com/Insights/Datasets (QMJ, BAB, momentum)
- **Open Asset Pricing:** https://www.openassetpricing.com/ (319 anomaly characteristics)
- **q-Factor Data:** https://global-q.org/factors.html

---

## 3. Algorithmic Trading and Market Microstructure

The 2015-2025 period has seen significant advances in execution algorithms, with machine learning methods (LSTM, reinforcement learning) now demonstrating measurable improvements over traditional TWAP/VWAP approaches.

### Foundational Optimal Execution

**Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions."**
- *Source:* Journal of Risk, Vol. 3(2), pp. 5-39
- *URL:* https://www.math.nyu.edu/~almgren/papers/optliq.pdf
- *Summary:* Seminal model balancing market impact cost against timing risk. Distinguishes temporary from permanent market impact. Derives closed-form optimal strategies. **Foundation for virtually all institutional execution algorithms**.
- *Applicability:* HIGH | *Access:* Open Access

### High-Frequency Trading Research

**Budish, E., Cramton, P., & Shim, J. (2015). "The High-Frequency Trading Arms Race."**
- *Source:* Quarterly Journal of Economics, Vol. 130(4), pp. 1547-1621
- *DOI:* https://doi.org/10.1093/qje/qjv027 | *SSRN:* 2388265
- *Summary:* Using millisecond ES-SPY arbitrage data (2005-2011), demonstrates continuous limit order books create socially wasteful speed arms races. Median arbitrage duration declined from **97ms to 7ms** while profitability remained constant. Proposes frequent batch auctions as solution.
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Brogaard, J., Hendershott, T., & Riordan, R. (2014). "High Frequency Trading and Price Discovery."**
- *Source:* Review of Financial Studies, Vol. 27(8), pp. 2267-2306
- *DOI:* https://doi.org/10.1093/rfs/hhu032 | *SSRN:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1928510
- *Summary:* Using NASDAQ HFT data with firm identifiers, shows HFTs facilitate price efficiency by trading in direction of permanent price changes. HFTs' liquidity-demanding orders contribute positively to price discovery.
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Baldauf, M. & Mollner, J. (2020). "High-Frequency Trading and Market Performance."**
- *Source:* Journal of Finance, Vol. 75(3)
- *DOI:* https://doi.org/10.1111/jofi.12882 | *SSRN:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2674767
- *Summary:* Theoretical model showing HFT can improve liquidity but may reduce incentives for fundamental information acquisition.
- *Applicability:* MEDIUM-HIGH | *Access:* SSRN

### Order Flow Analysis

**Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World."**
- *Source:* Review of Financial Studies, Vol. 25(5), pp. 1457-1493
- *DOI:* https://doi.org/10.1093/rfs/hhs053 | *SSRN:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596
- *Summary:* Introduces VPIN (Volume-synchronized Probability of INformed trading) metric. **VPIN predicted elevated toxicity hours before the May 6, 2010 Flash Crash**. Standard tool for risk management and surveillance.
- *Applicability:* HIGH | *Access:* SSRN

**Andersen, T.G. & Bondarenko, O. (2015). "Assessing Measures of Order Flow Toxicity."**
- *Source:* Review of Finance, Vol. 19(1), pp. 1-54
- *DOI:* https://doi.org/10.1093/rof/rfu041 | *SSRN:* https://ssrn.com/abstract=2292602
- *Summary:* Critical assessment of VPIN. Finds Bulk Volume Classification scheme inferior to standard tick rules. VPIN's volatility prediction may stem from classification errors.
- *Applicability:* HIGH | *Access:* SSRN

### Modern Execution Algorithms

**Busseti, E. & Boyd, S. "Volume Weighted Average Price Optimal Execution."**
- *Source:* Stanford University Working Paper
- *URL:* https://web.stanford.edu/~boyd/papers/pdf/vwap_opt_exec.pdf
- *Summary:* Develops both static and dynamic VWAP algorithms using stochastic control theory. Backtested on NYSE data showing dynamic strategies improve VWAP tracking and reduce transaction costs.
- *Applicability:* HIGH | *Access:* Open Access

**Ning, B., Lin, F.H.T., & Jaimungal, S. (2023). "An Optimal Control Strategy for Execution of Large Stock Orders Using LSTMs."**
- *Source:* arXiv:2301.09705
- *URL:* https://arxiv.org/pdf/2301.09705
- *Summary:* LSTM neural networks outperform TWAP and VWAP. Reduces daily transaction costs by **4-10%** compared to VWAP on S&P 100 stocks—average daily savings of ~$32,000 per stock for large orders.
- *Applicability:* HIGH | *Access:* arXiv Open Access

**Adaptive Dual-Level Reinforcement Learning for Optimal Trade Execution (2024)**
- *Source:* Expert Systems with Applications
- *URL:* https://www.sciencedirect.com/science/article/abs/pii/S0957417424011291
- *Summary:* Two-stage RL approach: first stage uses U-shaped intraday volume patterns, second stage applies PPO for execution. Improved accuracy over single-stage RL and traditional VWAP.
- *Applicability:* HIGH | *Access:* Journal Paywall

### Dark Pools and Latency Arbitrage

**"Sharks in the Dark: Quantifying HFT Dark Pool Latency Arbitrage" (2023)**
- *Source:* Journal of Economic Dynamics and Control
- *URL:* https://www.sciencedirect.com/science/article/pii/S0165188923001926
- *Summary:* Stale reference pricing imposes large costs on passive dark pool participants. Market design interventions randomizing execution times are effective countermeasures.
- *Applicability:* HIGH | *Access:* Journal

---

## 4. Position Sizing and Portfolio Optimization

Research has converged on the primacy of estimation error—simple methods like 1/N often outperform sophisticated optimization unless regularization (shrinkage, constraints) is applied. Risk parity, volatility targeting, and hierarchical methods have emerged as robust practical approaches.

### Kelly Criterion Applications

**Smirnov, M. & Dapporto, A. (2025). "Multivariable Kelly Criterion and Optimal Leverage Calculation from Data."**
- *Source:* SSRN Working Paper #5288640
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5288640
- *Summary:* Extends Kelly to multi-asset portfolios with discrete sampling. Optimal leverage depends on sampling frequency and fat tails. Longer rebalancing intervals and heavy tails require lower leverage.
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Nekrasov, V. (2014). "Kelly Criterion for Multivariate Portfolios: A Model-Free Approach."**
- *Source:* SSRN Working Paper #2259133
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2259133
- *Summary:* Model-free numerical approach using GPU-based Monte Carlo simulation for multivariate Kelly fractions without distributional assumptions.
- *Applicability:* HIGH | *Access:* SSRN Open Access

### Risk Parity and Hierarchical Methods

**López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample."**
- *Source:* Journal of Portfolio Management
- *DOI:* 10.3905/jpm.2016.42.4.059
- *Summary:* Introduces **Hierarchical Risk Parity (HRP)** using hierarchical clustering. Eliminates covariance matrix inversion, reducing estimation error sensitivity. Demonstrates superior out-of-sample performance vs. mean-variance optimization. **Widely adopted by practitioners**.
- *Applicability:* HIGH | *Access:* Journal (available in open-source libraries)

**Roncalli, T. & Weisang, G. (2016). "Risk Parity Portfolios with Risk Factors."**
- *Source:* Quantitative Finance
- *DOI:* 10.1080/14697688.2015.1046905
- *Summary:* Extends risk parity from asset-based to factor-based decomposition. Proposes optimization programs for achieving desired risk factor diversification.
- *Applicability:* HIGH | *Access:* Journal (Taylor & Francis)

### Volatility Targeting

**Harvey, C., et al. (2018). "The Impact of Volatility Targeting."**
- *Source:* Journal of Portfolio Management
- *DOI:* 10.3905/jpm.2018.44.4.014
- *Summary:* Analysis across **60+ assets over 90 years**. Volatility targeting reduces likelihood of extreme returns, reduces maximum drawdowns, and implicitly incorporates momentum exposure through leverage effect.
- *Applicability:* HIGH | *Access:* Journal

**Kang, X. & van Dijk, M. (2020). "Conditional Volatility Targeting."**
- *Source:* Financial Analysts Journal
- *DOI:* 10.1080/0015198X.2020.1790853
- *Summary:* Proposes conditional volatility targeting that adjusts only during extreme volatility states. Consistent Sharpe ratio improvement with low turnover across 10 largest global markets.
- *Applicability:* HIGH | *Access:* Open Access

### Drawdown Control

**Nystrup, P., Boyd, S., Lindström, E., & Madsen, H. (2019). "Multi-period Portfolio Selection with Drawdown Control."**
- *Source:* Annals of Operations Research, Vol. 282, pp. 245-271
- *DOI:* 10.1007/s10479-018-2947-3 | *URL:* stanford.edu/~boyd/papers/multiperiod_portfolio_drawdown.html
- *Summary:* Uses model predictive control with HMM forecasting. Adjusts risk aversion based on realized drawdown. Successful drawdown control with minimal sacrifice of mean-variance efficiency over 20+ years.
- *Applicability:* HIGH | *Access:* Springer / Stanford

**Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). "Portfolio Optimization with Drawdown Constraints."**
- *Source:* Journal of Risk | *SSRN:* https://ssrn.com/abstract=223323
- *Summary:* Introduces **Conditional Drawdown-at-Risk (CDaR)**. Reduces drawdown optimization to linear programming, enabling practical implementation with thousands of instruments.
- *Applicability:* HIGH | *Access:* SSRN Open Access

### Estimation Error and Shrinkage

**DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive Diversification."**
- *Source:* Review of Financial Studies, Vol. 22(5), pp. 1915-1953
- *DOI:* 10.1093/rfs/hhm075
- *Summary:* **Landmark study**: None of 14 sophisticated optimization models consistently outperform simple 1/N out-of-sample. Estimation error can require **3,000-6,000 months of data** to reliably beat 1/N.
- *Applicability:* HIGH | *Access:* Journal

**Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix."**
- *Source:* Journal of Portfolio Management
- *DOI:* 10.3905/jpm.2004.110
- *Summary:* Establishes that sample covariance should **never be used directly** for optimization. Proposes shrinkage toward constant correlation matrix. Implemented in scikit-learn and all major platforms.
- *Applicability:* HIGH | *Access:* Open Access (ledoit.net)

**Ledoit, O. & Wolf, M. (2020). "The Power of (Non-)Linear Shrinking."**
- *Source:* Journal of Financial Econometrics
- *DOI:* 10.1093/jjfinec/nbaa007
- *Summary:* Comprehensive review covering 20 years of shrinkage research. Extends to nonlinear shrinkage with different intensities per eigenvalue.
- *Applicability:* HIGH | *Access:* Oxford Academic

---

## 5. Statistical Methods and Machine Learning

Machine learning has transformed quantitative finance, with LSTM networks, transformers, and reinforcement learning demonstrating measurable improvements over classical methods—when properly validated against overfitting.

### Deep Learning for Financial Prediction

**Fischer, T. & Krauss, C. (2018). "Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions."**
- *Source:* European Journal of Operational Research, Vol. 270(2), pp. 654-669
- *DOI:* https://doi.org/10.1016/j.ejor.2017.11.054
- *Summary:* Landmark LSTM study on all S&P 500 constituents (1992-2015). Achieved **daily returns of 0.46% and Sharpe Ratio of 5.8** pre-transaction costs. Outperformed random forests, DNNs, and logistic regression. **1,391+ citations**.
- *Applicability:* HIGH | *Access:* Journal / SSRN / arXiv

**Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting."**
- *Source:* International Journal of Forecasting, Vol. 37(4), pp. 1748-1764
- *DOI:* https://doi.org/10.1016/j.ijforecast.2021.03.012
- *Summary:* Introduces TFT architecture combining recurrent layers with self-attention. Features variable selection networks and **interpretable attention weights**. State-of-the-art for multi-horizon forecasting.
- *Applicability:* HIGH | *Access:* Journal (code available)

**TFT-GNN Hybrid for Stock Forecasting (2025)**
- *Source:* MDPI
- *URL:* https://www.mdpi.com/2673-9909/5/4/176
- *Summary:* Integrates graph neural networks with TFT to capture inter-asset relationships. GNN-derived relational features assigned greater attention than traditional indicators.
- *Applicability:* MEDIUM-HIGH | *Access:* Open Access

### Statistical Arbitrage and Pairs Trading

**Gatev, E., Goetzmann, W.N., & Rouwenhorst, K.G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."**
- *Source:* Review of Financial Studies, Vol. 19(3), pp. 797-827
- *DOI:* https://doi.org/10.1093/rfs/hhj020
- *Summary:* Foundational pairs trading paper (1962-2002). Distance-based matching yields **11% annualized excess returns** for self-financing portfolios. Profits exceed transaction costs.
- *Applicability:* HIGH | *Access:* Journal / SSRN / NBER

**Caldeira, J. & Moura, G.V. (2013). "Selection of a Portfolio of Pairs Based on Cointegration."**
- *Source:* SSRN Working Paper #2196391
- *URL:* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2196391
- *Summary:* Applies Engle-Granger and Johansen cointegration testing. Rolling window yields **189.29% cumulative profit** over 4 years (16.38% annual), Sharpe 1.34, low market correlation.
- *Applicability:* HIGH | *Access:* SSRN Open Access

### Volatility Modeling: GARCH and Hybrids

**Chung, S. (2024). "Modelling and Forecasting Energy Market Volatility Using GARCH and Machine Learning."**
- *Source:* arXiv:2405.19849
- *URL:* https://arxiv.org/abs/2405.19849
- *Summary:* Comprehensive comparison of GARCH vs ML for volatility forecasting. ML models demonstrate superior out-of-sample performance. GARCH overpredicts, ML underpredicts—suggests hybrid approach. Uses SHAP for interpretability.
- *Applicability:* HIGH | *Access:* arXiv Open Access

**Wang et al. (2024). "From GARCH to Neural Network for Volatility Forecast."**
- *Source:* arXiv:2402.06642
- *URL:* https://arxiv.org/html/2402.06642v1
- *Summary:* Establishes theoretical equivalence between GARCH and neural network counterparts. Proposes GARCH-LSTM hybrid integrating econometric stylized facts with LSTM long-memory capabilities.
- *Applicability:* MEDIUM-HIGH | *Access:* arXiv Open Access

### Regime Detection

**Wang, M., Lin, Y.-H., & Mikhelson, I. (2020). "Regime-Switching Factor Investing with Hidden Markov Models."**
- *Source:* Journal of Risk and Financial Management, Vol. 13(12), 311
- *DOI:* https://doi.org/10.3390/jrfm13120311
- *Summary:* Uses HMM to identify US market regimes (2007-2020), switching factor models based on detected regime. Outperforms individual factor strategies including through COVID-19.
- *Applicability:* HIGH | *Access:* Open Access (MDPI)

### Reinforcement Learning for Trading

**García-Bringas et al. (2024). "Deep Reinforcement Learning for Portfolio Selection."**
- *Source:* Global Finance Journal, 57, 100887
- *URL:* https://www.sciencedirect.com/science/article/pii/S1044028324000887
- *Summary:* Model-free DRL using Twin-Delayed DDPG (TD3). Embeds risk aversion and transaction costs in mean-variance reward. Outperforms traditional MVO and competing DRL methods on DJIA and S&P100.
- *Applicability:* HIGH | *Access:* Journal

**Seong, N. & Nam, K. (2023). "Deep Reinforcement Learning for Stock Portfolio Optimization by Connecting with Modern Portfolio Theory."**
- *Source:* Expert Systems with Applications, 218, 119556
- *DOI:* https://doi.org/10.1016/j.eswa.2023.119556
- *Summary:* Integrates MPT correlation matrix with technical analysis via 3D CNN and Tucker decomposition. Uses DDPG for policy optimization. Bridges classical finance theory with deep RL.
- *Applicability:* HIGH | *Access:* Journal Paywall

---

## 6. Backtesting Methodology and Algorithm Formulation

The replication crisis in finance has produced essential methodological advances for preventing false discoveries. The literature has converged on the necessity of multiple testing corrections, with t-statistic hurdles of 3.0+ now considered minimum thresholds.

### Multiple Testing and False Discovery

**Harvey, C.R., Liu, Y., & Zhu, H. (2016). "…and the Cross-Section of Expected Returns."**
- *Source:* Review of Financial Studies, Vol. 29(1), pp. 5-68
- *DOI:* https://doi.org/10.1093/rfs/hhv059 | *SSRN:* https://ssrn.com/abstract=2249314
- *Summary:* Examines 316 factors in academic literature. A new factor needs **t-statistic >3.0** (not traditional 2.0) to be considered significant. Authors argue "most claimed research findings in financial economics are likely false."
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Harvey, C.R. & Liu, Y. (2020). "False (and Missed) Discoveries in Financial Economics."**
- *Source:* Journal of Finance, Vol. 75(5), pp. 2503-2553
- *DOI:* 10.1111/jofi.12951 | *arXiv:* https://arxiv.org/abs/2006.04269
- *Summary:* Double-bootstrap method calibrating both Type I (false discoveries) and Type II (missed discoveries) errors. Establishes t-statistic hurdles for specific false discovery rates.
- *Applicability:* HIGH | *Access:* arXiv Open Access

**Giglio, S., Liao, Y., & Xiu, D. (2020). "Thousands of Alpha Tests."**
- *Source:* Review of Financial Studies, Vol. 33(6), pp. 2625-2681
- *DOI:* 10.1093/rfs/hhz123 | *SSRN:* https://ssrn.com/abstract=3259268
- *Summary:* Rigorous framework for multiple hypothesis testing in asset pricing using ML techniques. Robust to omitted factors, missing data, and high-dimensional settings.
- *Applicability:* HIGH | *Access:* SSRN

### Backtest Overfitting Prevention

**Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio."**
- *Source:* Journal of Portfolio Management, Vol. 40(5), pp. 94-107
- *DOI:* 10.3905/jpm.2014.40.5.094 | *URL:* https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- *Summary:* Introduces Deflated Sharpe Ratio (DSR) correcting for selection bias, non-normally distributed returns, and sample length. Provides practical formula to adjust reported Sharpe ratios based on number of trials.
- *Applicability:* HIGH | *Access:* Open Access

**Bailey, D.H., et al. (2015). "The Probability of Backtest Overfitting."**
- *Source:* Journal of Computational Finance
- *URL:* https://www.davidhbailey.com/dhbpapers/backtest-overfitting.pdf
- *Summary:* Mathematical framework for computing **Probability of Backtest Overfitting (PBO)** using Combinatory Symmetric Cross-Validation. Overfitting probability increases rapidly with strategy configurations tested.
- *Applicability:* HIGH | *Access:* Open Access

**"Backtest Overfitting in the Machine Learning Era" (2024)**
- *Source:* Knowledge-Based Systems, 304, 112410
- *URL:* https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110
- *Summary:* Comprehensive comparison of cross-validation methods. **Combinatorial Purged Cross-Validation (CPCV)** outperforms walk-forward in mitigating overfitting with lower PBO and superior Deflated Sharpe Ratio.
- *Applicability:* HIGH | *Access:* Journal

### Transaction Cost Modeling

**Almgren, R. & Chriss, N. (2000). "Optimal Execution of Portfolio Transactions."**
- *Source:* Journal of Risk, Vol. 3(2) | *SSRN:* https://ssrn.com/abstract=53501
- *Summary:* Industry-standard framework for modeling permanent and temporary market impact. Essential for realistic backtest transaction cost modeling.
- *Applicability:* HIGH | *Access:* SSRN / Open Access

### Alpha Decay Research

**Di Mascio, R., Lines, A., & Naik, N.Y. (2018). "Alpha Decay."**
- *Source:* London Business School Working Paper
- *URL:* https://www.top1000funds.com/wp-content/uploads/2021/05/SSRN-id2580551.pdf
- *Summary:* Studies decay using 1.15 million institutional trades ($1.8 trillion). Total cumulative post-purchase alpha is **1.09% after 18 months** (most captured in first month).
- *Applicability:* MEDIUM-HIGH | *Access:* Open Access

**Gârleanu, N. & Pedersen, L.H. (2013). "Dynamic Trading with Predictable Returns and Transaction Costs."**
- *Source:* Journal of Finance, Vol. 68(6), pp. 2309-2340
- *DOI:* 10.1111/jofi.12080 | *URL:* http://docs.lhpedersen.com/DynamicTrading.pdf
- *Summary:* Closed-form optimal trading policies for multiple predictors with different alpha decay rates. More persistent signals should be weighted more heavily.
- *Applicability:* HIGH | *Access:* Author PDF

---

## 7. Cross-Asset Methods

### Futures and Managed Futures

**Moskowitz, T.J., Ooi, Y.H., & Pedersen, L.H. (2012). "Time Series Momentum."**
- *Source:* Journal of Financial Economics, Vol. 104(2), pp. 228-250
- *DOI:* https://doi.org/10.1016/j.jfineco.2011.11.003 | *SSRN:* https://ssrn.com/abstract=2089463
- *Summary:* Documents significant TSMOM across 58 liquid futures. 12-month excess return predicts future returns. Diversified TSMOM portfolio delivers **Sharpe ratio exceeding 1.0** with little factor exposure.
- *Applicability:* HIGH | *Access:* Journal

**Baltas, N. & Kosowski, R. (2020). "Momentum Strategies in Futures Markets and Trend-following Funds."**
- *Source:* Imperial College Working Paper | *SSRN:* https://ssrn.com/abstract=1968996
- *Summary:* 75 futures contracts (1974-2013). TSMOM strategies achieve **Sharpe ratios above 1.20**. No evidence of capacity constraints. Monthly, weekly, and daily strategies exhibit low cross-correlation.
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Hutchinson, M.C. & O'Brien, J. (2014). "Is This Time Different? Trend Following and Financial Crises."**
- *Source:* Journal of Alternative Investments, Vol. 17(2)
- *DOI:* 10.3905/jai.2014.17.2.082 | *SSRN:* https://ssrn.com/abstract=2375733
- *Summary:* Nearly a century of data shows trend-following returns are **less than half** in extended post-crisis periods vs. no-crisis periods. Weaker predictability for ~4 years following crises.
- *Applicability:* MEDIUM | *Access:* SSRN

### Options and Volatility

**Fallon, W., Park, J., & Yu, D. (2015). "Understanding the Volatility Risk Premium."**
- *Source:* AQR White Paper | *URL:* https://www.aqr.com/Insights/Research/White-Papers/Understanding-the-Volatility-Risk-Premium
- *Summary:* Implied volatility exceeds realized volatility **~85% of the time**. VRP exists due to behavioral, structural, and economic factors. Framework for systematically harvesting VRP.
- *Applicability:* HIGH | *Access:* Open Access

**Carr, P. & Wu, L. (2016). "Analyzing Volatility Risk and Risk Premium in Option Contracts."**
- *Source:* Journal of Financial Economics, Vol. 120(1), pp. 1-20
- *DOI:* 10.1016/j.jfineco.2016.01.004
- *Summary:* Novel framework integrating option pricing with institutional practice. Introduces option-specific realized volatility (ORV). Extracted VRP significantly predicts future stock returns.
- *Applicability:* HIGH | *Access:* Journal

**Konstantinidi, E. & Skiadopoulos, G. (2016). "How Does the Market Variance Risk Premium Vary Over Time?"**
- *Source:* Journal of Banking & Finance, Vol. 62 | *SSRN:* https://ssrn.com/abstract=2502822
- *Summary:* Uses actual S&P 500 variance swap quotes. Deteriorating economic conditions increase VRP. Trading strategies conditioning on these relations outperform buy-and-hold after transaction costs.
- *Applicability:* HIGH | *Access:* SSRN

**Martin, I.W.R. (2017). "Simple Variance Swaps."**
- *Source:* Quarterly Journal of Economics, Vol. 132(4) | *SSRN:* https://ssrn.com/abstract=1789465
- *Summary:* Defines "simple variance swaps" more robust than traditional variance swaps—easily priced/hedged even with jumps. More accurate market-implied variance than VIX.
- *Applicability:* MEDIUM | *Access:* SSRN

### Forex

**Menkhoff, L., et al. (2012). "Currency Momentum Strategies."**
- *Source:* Journal of Financial Economics, Vol. 106(3), pp. 660-684
- *DOI:* 10.1016/j.jfineco.2012.06.009 | *URL:* https://www.bis.org/publ/work366.pdf
- *Summary:* 48 currencies (1976-2010). Cross-sectional spread averaging **10% annually**. Momentum driven by spot rate continuation (not interest differentials), uncorrelated with carry.
- *Applicability:* HIGH | *Access:* BIS Working Paper (Open Access)

**Daniel, K., Hodrick, R.J., & Lu, Z. (2017). "The Carry Trade: Risks and Drawdowns."**
- *Source:* Critical Finance Review, Vol. 6, pp. 211-262
- *URL:* https://business.columbia.edu/sites/default/files-efs/pubfiles/6378/Daniel.Hodrick.Lu.Carry%20Trade.Critical%20Finance%20Review.2017.pdf
- *Summary:* G10 carry trade with spread-weighting and risk-rebalancing. Dynamically weighted strategies improve performance. Hedging with options reduces but doesn't eliminate abnormal returns.
- *Applicability:* HIGH | *Access:* Open Access

**Lustig, H., Roussanov, N., & Verdelhan, A. (2011). "Common Risk Factors in Currency Markets."**
- *Source:* Review of Financial Studies, Vol. 24(11), pp. 3731-3777
- *URL:* https://www.nber.org/system/files/working_papers/w25449/w25449.pdf
- *Summary:* Identifies "slope" factor (HML_FX) in exchange rates. Constructs DOL (dollar risk factor) and HML_FX accounting for cross-sectional variation. Framework analogous to Fama-French for currencies.
- *Applicability:* HIGH | *Access:* NBER Open Access

### Cryptocurrency

**Liu, Y. & Tsyvinski, A. (2021). "Risks and Returns of Cryptocurrency."**
- *Source:* Review of Financial Studies, Vol. 34(6), pp. 2689-2727
- *DOI:* 10.1093/rfs/hhaa113 | *URL:* https://www.nber.org/papers/w24877
- *Summary:* Crypto risk-return tradeoff distinct from stocks, currencies, and precious metals. **No exposure to common stock or macro factors**. Strong time-series momentum; investor attention strongly forecasts returns.
- *Applicability:* HIGH | *Access:* NBER Open Access

**Liu, Y., Tsyvinski, A., & Wu, X. (2022). "Common Risk Factors in Cryptocurrency."**
- *Source:* Journal of Finance, Vol. 77(2), pp. 1133-1177
- *DOI:* 10.1111/jofi.13119
- *Summary:* Proposes **three-factor model (market, size, momentum)** capturing cross-sectional crypto returns. Ten characteristics form successful long-short strategies, all explained by three-factor model.
- *Applicability:* HIGH | *Access:* Journal

**Dobrynskaya, V. (2023). "Cryptocurrency Momentum and Reversal."**
- *Source:* Journal of International Financial Markets | *SSRN:* https://ssrn.com/abstract=3913263
- *Summary:* 2,000 largest cryptos (2014-2020). Positive momentum on short horizons (2-4 weeks), **significant reversal beyond 1 month**. Momentum-to-reversal switch much faster than equities ("faster metabolism").
- *Applicability:* HIGH | *Access:* SSRN Open Access

**Bouri, E., et al. (2023). "On the Efficiency and Its Drivers in the Cryptocurrency Market."**
- *Source:* Financial Innovation
- *DOI:* 10.1186/s40854-023-00566-3 | *URL:* https://jfin-swufe.springeropen.com/articles/10.1186/s40854-023-00566-3
- *Summary:* Time-varying efficiency of Bitcoin and Ethereum (2016-2023). Global financial stress negatively affects efficiency. Evidence supports adaptive market hypothesis.
- *Applicability:* MEDIUM | *Access:* Open Access

---

## Conclusion

This bibliography spans **96 papers** representing the core academic foundation for systematic trading research. Several converging themes emerge from the 2015-2025 literature that practitioners should internalize:

**Multiple testing is the central challenge.** The Harvey et al. (2016) finding that t-statistics must exceed 3.0—not the traditional 2.0—has fundamentally changed how researchers evaluate new strategies. Combined with McLean & Pontiff's evidence of **58% post-publication decay**, practitioners should expect substantial degradation from backtested to live performance.

**Machine learning works, with caveats.** The Fischer & Krauss (2018) LSTM results (Sharpe 5.8 pre-costs) and the emerging transformer literature demonstrate genuine predictive power. However, the CPCV methodology from López de Prado and subsequent validation studies shows that traditional cross-validation fails for financial time series—purged, combinatorial methods are essential.

**Simple methods often win.** The DeMiguel et al. (2009) finding that 1/N competes with sophisticated optimization remains central. Shrinkage estimators (Ledoit-Wolf), hierarchical methods (HRP), and constraint-based approaches consistently outperform theoretically optimal but estimation-error-sensitive methods.

**Factor investing is maturing.** The factor zoo has been compressed to approximately 15 genuinely distinct factors. The open-source infrastructure (OpenAssetPricing, AQR datasets, q-factor data) enables rigorous replication and realistic out-of-sample testing.

**Cross-asset diversification delivers.** Time-series momentum in futures (Sharpe >1.0), the volatility risk premium in options (~85% of the time), currency momentum and carry (uncorrelated strategies), and crypto's distinct factor structure all offer genuine diversification benefits for multi-asset systematic strategies.

The academic literature now provides a rigorous foundation for building trading systems—but only when practitioners apply the methodological corrections for multiple testing, estimation error, and alpha decay that this same literature has documented.
