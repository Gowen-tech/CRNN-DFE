clc;
clear all;
close all;
a = [1-2j,2+3j,3-2j,4-0.5j,5+1j];
b = [1+2j,2-1j,3];
x=filter(b,1,a);
disp(x);
