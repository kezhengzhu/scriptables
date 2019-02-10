t = 0:0.002:(2000000*0.002);

% Plotting script, use x and y
scrsz = get(groot,'ScreenSize');
figure('Position',[scrsz(4)/6 1 scrsz(3)*5/6 scrsz(4)]);

sp{1} = subplot(2,2,1);


sp{2} = subplot(2,2,2);

plot(t, eta1, '--', 'LineWidth', 0.8)
plot(t, eta1_ave, 'k-', 'LineWidth', 1.5)

sp{3} = subplot(2,2,3);


sp{4} = subplot(2,2,4);