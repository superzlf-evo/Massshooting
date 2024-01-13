<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" >

<title>某国各州大型枪支暴力事件预测系统 | superzlf-evo</title>

\



    <meta name="description" content="

背景与目的
众所周知，霉国犯罪事件频发，尤其枪击事件类型。所以该系统主要是分析该国每年各州的枪击犯罪事件，通过深度学习去预测大型枪支暴力事件。

方式
在GVA，BEA，Kaggle等网站上获取霉国各州10年内数据作为数据集，对相关数据..." />
    <meta name="keywords" content="python" />
  </head>
  <body>
    <div id="app" class="main">

      <div class="sidebar" :class="{ 'full-height': menuVisible }">
  <div class="top-container" data-aos="fade-right">
    <div class="top-header-container">
      <a class="site-title-container" href="https://superzlf-evo.github.io">
        <img src="https://superzlf-evo.github.io/images/avatar.png?v=1705114418750" class="site-logo">
        <h1 class="site-title">superzlf-evo</h1>
      </a>
      <div class="menu-btn" @click="menuVisible = !menuVisible">
        <div class="line"></div>
      </div>
    </div>
    <div>
      
        
          <a href="/" class="site-nav">
            首页
          </a>
        
      
        
          <a href="/archives" class="site-nav">
            归档
          </a>
        
      
        
          <a href="/tags" class="site-nav">
            标签
          </a>
        
      
        
          <a href="/post/about" class="site-nav">
            关于
          </a>
        
      
    </div>
  </div>
  <div class="bottom-container" data-aos="flip-up" data-aos-offset="0">
    <div class="social-container">
      
        
      
        
      
        
      
        
      
        
      
    </div>
    <div class="site-description">
      就这样
    </div>
    <div class="site-footer">
       | <a class="rss" href="https://superzlf-evo.github.io/atom.xml" target="_blank">RSS</a>
    </div>
  </div>
</div>


      <div class="main-container">
        <div class="content-container" data-aos="fade-up">
          <div class="post-detail">
            <h2 class="post-title">某国各州大型枪支暴力事件预测系统</h2>
            <div class="post-date">2023-10-01</div>
            
              <div class="feature-container" style="background-image: url('https://superzlf-evo.github.io/post-images/mei-guo-ge-zhou-da-xing-qiang-zhi-bao-li-shi-jian-yu-ce-xi-tong.png')">
              </div>
            
            <div class="post-content" v-pre>
              <!-- more -->
<!-- more -->
<h1 id="背景与目的">背景与目的</h1>
<p>众所周知，霉国犯罪事件频发，尤其枪击事件类型。所以该系统主要是分析该国每年各州的枪击犯罪事件，通过深度学习去预测大型枪支暴力事件。</p>
<hr>
<h1 id="方式">方式</h1>
<p>在GVA，BEA，Kaggle等网站上获取霉国各州10年内数据作为数据集，对相关数据进行可视化的热力图，折线图，柱状图的展示。<br>
使用每年季度，霉国各州，人口，失业率，收入，人均GDP信息，对2023各州4个季度的犯罪事件数量进行预测。</p>
<hr>
<h1 id="数据收集">数据收集</h1>
<p>霉国2014-2022各州枪击事件数量数据：<br>
www.gunviolencearchive.org/past-tolls<br>
霉国2014-2022各州季度国内生产总值数据：<br>
www.bea.gov<br>
霉国2014-2022各州人均个人收入数据：<br>
www.bea.gov<br>
霉国2014-2022各州人口数量分布数据：<br>
www.census.gov<br>
霉国2014-2022各州人口失业率分布数据：<br>
www.kaggle.com<br>
www.macroview.club<br>
cn.investing.com</p>
<hr>
<h1 id="详细实现">详细实现</h1>
<hr>
<h2 id="1数据可视化分析">1.数据可视化分析</h2>
<h3 id="收入">收入</h3>
<p>使用热力图和折线图显示该国枪击犯罪事件：<br>
<img src="https://superzlf-evo.github.io/post-images/1705073860104.png" alt="" loading="lazy"></p>
<figure data-type="image" tabindex="1"><img src="https://superzlf-evo.github.io/post-images/1705074194758.png" alt="" loading="lazy"></figure>
<hr>
<h3 id="人口">人口</h3>
<figure data-type="image" tabindex="2"><img src="https://superzlf-evo.github.io/post-images/1705074357200.png" alt="" loading="lazy"></figure>
<figure data-type="image" tabindex="3"><img src="https://superzlf-evo.github.io/post-images/1705074419903.png" alt="" loading="lazy"></figure>
<hr>
<h3 id="gdp">GDP</h3>
<figure data-type="image" tabindex="4"><img src="https://superzlf-evo.github.io/post-images/1705074516893.png" alt="" loading="lazy"></figure>
<figure data-type="image" tabindex="5"><img src="https://superzlf-evo.github.io/post-images/1705074656108.png" alt="" loading="lazy"></figure>
<hr>
<h3 id="失业率">失业率</h3>
<figure data-type="image" tabindex="6"><img src="https://superzlf-evo.github.io/post-images/1705074768486.png" alt="" loading="lazy"></figure>
<figure data-type="image" tabindex="7"><img src="https://superzlf-evo.github.io/post-images/1705074834874.png" alt="" loading="lazy"></figure>
<hr>
<h2 id="全神经网络模型预测">全神经网络模型预测</h2>
<p>将收集2014年到2023年美国各州的人口数量，失业率，GDP，收入等数据，并且通过美国GVA网站收集2014年到2022年的该国各州的大规模枪支暴力事件。我们需要从2014年到2022年的相关数据去预测2023年该国各州的大规模枪支暴力事件数量。</p>
<h3 id="标准化数据">标准化数据</h3>
<p>项目收集了美国各州的人口数量，失业率，GDP，收入，和犯罪事件，将各数据合并为一个表，并且去除空值。我们把时间划分为每年四个季度并且编码映射为数字，美国各州编码映射为数字，人口数量以万为单位，失业率去百分号，收入以千为单位，GDP以千为单位，收入以千为单位，累加同一时间段同一州的犯罪事件为数量。取特征为Year-Quarter，State，Population，EM_Rate，Income，GDP，Count。</p>
<h3 id="定义模型">定义模型</h3>
<p>我们需要选择合适的深度学习模型对大规模枪支暴力事件数量进行预测，最终我们决定使用全连接神经网络（Fully Connected Neural Network）模型去实现功能。全连接神经网络是一种基础的神经网络模型，其每个神经元与下一层的每个神经元都相连，结构相对简单。全连接神经网络的学习和表达能力较强，适用于解决模式识别、数据分类和回归等问题。其基本原理是通过反向传播算法，不断调整神经网络的权重和偏置，使得网络的输出与目标输出之间的误差尽可能小。全连接网络是一种基本的神经网络结构，通过将输入数据映射到隐藏层，再经过非线性变换映射到输出层，从而实现对数据的分类或回归预测。</p>
<pre><code class="language-python">model = Sequential()
model.add(Dense(6, input_dim=6, activation='PReLU'))  # 输入层
model.add(Dense(128, activation='PReLU'))  # 隐藏层
model.add(Dense(1, activation='PReLU'))  # 输出层
model.compile(loss='mean_squared_error', optimizer='adam')  # 使用均方误差作为损失函数，优化器为Adam
</code></pre>
<hr>
<h3 id="训练数据">训练数据</h3>
<p>对于测试模型的准确性，将数据集以9:1 的形式随机划分为训练集和测试集：x_tran,y_tran,x_test,y_text。通过大量迭代和选取不同样本大小训练训练集，使训练过程中损失函数的值越小。由于通过模型输出的数值y_pred为浮点型，最终需要对数值整形化并且对负值归零处理。所以同时定义了三个判断准确率方法分别得出：accuracy，accuracy_1，accuracy_2;accuracy为预测值数列y_pred占测试集y_test中的准确率；accuracy1为预测值数列y_pred与y_test误差值在1以内情况下占测试集y_test中的准确率；accuracy2为预测值数列y_pred与y_test误差值在2以内情况下占测试集y_test中的准确率；<br>
经过多次模型训练测试，在输入样本大小为128，迭代次数为5000次，输入层6节点，隐藏层128个节点时，测试结果最优，损失函数数值趋于1.0，accuracy为0.42，accuracy1为0.76，accuracy2为0.89.我们保存并使用该模型去预测2023年美国各州的大规模枪击事件。</p>
<hr>
<h3 id="结果展示">结果展示</h3>
<p>通过14到22年的数据预测得出23年4个季度的各州犯罪时间数,取预测的前3季度和真实23年数据对比:<br>
<img src="https://superzlf-evo.github.io/post-images/1705077851315.png" alt="" loading="lazy"></p>
<hr>
<h1 id="代码">代码</h1>
<p><a href="https://github.com/superzlf-evo/Massshooting">代码链接</a></p>
<hr>

            </div>
            
              <div class="tag-container">
                
                  <a href="https://superzlf-evo.github.io/tag/h8SEnBtoV/" class="tag">
                    python
                  </a>
                
              </div>
            
            

            
              
                <div id="gitalk-container" data-aos="fade-in"></div>
              

              
            

          </div>

        </div>
      </div>
    </div>

    <script src="https://unpkg.com/aos@next/dist/aos.js"></script>
<script type="application/javascript">

AOS.init();

var app = new Vue({
  el: '#app',
  data: {
    menuVisible: false,
  },
})

</script>





  
    <script src="https://unpkg.com/gitalk/dist/gitalk.min.js"></script>
    <script>

      var gitalk = new Gitalk({
        clientID: '98e722c8eede4cb43002',
        clientSecret: '35fca1d7567720e4cbecea65706eabd62596befd',
        repo: 'https://superzlf-evo.github.io',
        owner: 'superzlf-evo',
        admin: ['superzlf-evo'],
        id: (location.pathname).substring(0, 49),      // Ensure uniqueness and length less than 50
        distractionFreeMode: false  // Facebook-like distraction free mode
      })

      gitalk.render('gitalk-container')

    </script>
  

  




  </body>
</html>
