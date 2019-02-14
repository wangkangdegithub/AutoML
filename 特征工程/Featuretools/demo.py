# -*- coding: utf-8 -*-
"""
@author: wangkang (*/ω＼*)
@file: demo.py
@time: 2019/1/31 11:26
@desc: featuretools使用
"""

import featuretools as ft

# Create new entityset
es = ft.EntitySet(id='clients')
es = es.entity_from_dataframe(entity_id='clients', dataframe=clients,
                              index='client_id', time_index='joined')
es = es.entity_from_dataframe(entity_id='loans', dataframe=loans,
                              index='loans_id', time_index='joined')

# 对于没有唯一索引的表：我们需要传入参数make_index = True并指定索引的名称。
# 另外，虽然featuretools会自动推断实体中每个列的数据类型，但我们可以通过将列类型的字典传递给参数variable_types来覆盖它。
es = es.entity_from_dataframe(entity_id='payments', dataframe=payments,
                              variable_types={'missed': ft.variable_types.Categorical},
                              make_index=True, index='payment_id', time_index='payment_date')
# 指定Table Relationships。在执行聚合时，我们按父变量对子表进行分组，并计算每个父表的子表的统计信息。
# 要在featuretools中形式化关系，只需指定将两个表链接在一起的变量。客户端和loans表通过client_id变量链接，贷款和付款与loan_id链接。
# 创建关系并将其添加到entityset的语法如下所示:

r_client_previous = ft.Relationship(es['clients']['client_id'], es['loans']['client_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_client_previous)

# Relationship between previous loans and previous payments
r_payments = ft.Relationship(es['loans']['loan_id'], es['payments']['loan_id'])

# Add the relationship to the entity set
es = es.add_relationship(r_payments)
"""
entityset现在包含三个实体(表)以及将这些实体链接在一起的关系。
在添加实体和形式化关系之后，entityset就完成了，可以开始创建特性了。
"""

# Feature Primitives
# Aggregations:分组聚合
# Transformations:列之间计算

# Create new features using specified primitives
features, feature_names = ft.dfs(entityset=es, target_entity='clients',
                                 agg_primitives=['mean', 'max', 'percent_true', 'last'],
                                 trans_primitives=['years', 'month', 'subtract', 'divide'])
# 此外，我们还可以让ft自动为我们选择特征
features, feature_names = ft.dfs(entityset=es, target_entity='clients', max_depth=2)

# 总结

# 特征灾难

# 与机器学习中的许多主题一样，使用featuretools进行自动化特性工程是一个基于简单思想的复杂概念。
# 使用entityset、实体和关系的概念，featuretools可以执行深度特性合成来创建新特性。深度功能合成反过来又将功能原语——聚合(在表之间的一对多关系中起作用)和转换(应用于单个表中的一个或多个列的函数)——堆积在一起，从多个表构建新功能。
