clc;clear all;close all;
%%导入数据
label=load('svmAMa.txt');
matri=load('SVMnew.txt');
matrix=matri(:,[1 2 3]);
% 随机产生训练集和测试集
n = randperm(size(matrix,1));
% 训练集――264个样本  80%
train_matrix = matrix(n(1:264),:);
train_label = label(n(1:264),:);
% 测试集――66个样本
test_matrix = matrix(n(265:end),:);
test_label = label(n(265:end),:);
%数据归一化
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';
Test_matrix = mapminmax('apply',test_matrix',PS);
Test_matrix = Test_matrix';
%SVM创建/训练(RBF核函数)
%寻找最佳c/g参数――交叉验证方法
[c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 70;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];%将参数均以字符串的形式体现
        cg(i,j) = libsvmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
%创建/训练SVM模型
model = libsvmtrain(train_label,Train_matrix,cmd);
%SVM仿真测试
[predict_label_1,accuracy_1,decision_values1] = libsvmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,decision_values2] = libsvmpredict(test_label,Test_matrix,model);
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];
%绘图
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('真实类别','预测类别')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集SVM预测结果对比(RBF核函数)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)
