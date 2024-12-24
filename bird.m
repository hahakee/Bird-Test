function varargout = bird(varargin)
% BIRD MATLAB code for bird.fig
%      BIRD, by itself, creates a new BIRD or raises the existing
%      singleton*.
%
%      H = BIRD returns the handle to a new BIRD or the handle to
%      the existing singleton*.
%
%      BIRD('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in BIRD.M with the given input arguments.
%
%      BIRD('Property','Value',...) creates a new BIRD or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before bird_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to bird_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help bird

% Last Modified by GUIDE v2.5 25-Dec-2024 03:01:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @bird_OpeningFcn, ...
                   'gui_OutputFcn',  @bird_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before bird is made visible.
function bird_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to bird (see VARARGIN)

% Choose default command line output for bird
handles.output = hObject;

% Initialize image data
handles.img = []; % To store the loaded image

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes bird wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = bird_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_loadImage.
function pushbutton_loadImage_Callback(hObject, eventdata, handles)
% 打开文件选择对话框
[filename, pathname] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp'}, '选择图像文件');

% 检查用户是否选择了文件
if filename ~= 0
    % 读取图像文件
    img = imread(fullfile(pathname, filename));
    handles.img = img; % 保存到 handles 结构
    
    % 显示图像
    axes(handles.axes_image); % 激活Axes
    imshow(img); % 显示图像
    
    % 更新 handles
    guidata(hObject, handles);
else
    % 弹出提示框
    msgbox('未选择任何图像文件。', '错误', 'error');
end


% --- Executes on button press in pushbutton_showHistogram.
function pushbutton_showHistogram_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if isempty(handles.img)
    msgbox('请先加载图像', '错误', 'error');
    return;
end

% 转换为灰度图像
if size(handles.img, 3) == 3
    gray_img = rgb2gray(handles.img);
else
    gray_img = handles.img;
end

% 显示灰度直方图
axes(handles.axes_histogram); % 激活显示直方图的Axes
imhist(gray_img); % 显示直方图


% --- Executes on button press in pushbutton_histEqual.
function pushbutton_histEqual_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if isempty(handles.img)
    msgbox('请先加载图像', '错误', 'error');
    return;
end

% 转换为灰度图像
if size(handles.img, 3) == 3
    gray_img = rgb2gray(handles.img);
else
    gray_img = handles.img;
end

% 进行直方图均衡化
equalized_img = histeq(gray_img);

% 显示均衡化后的图像
axes(handles.axes_image);
imshow(equalized_img);

% 显示均衡化后的直方图
axes(handles.axes_histogram);
imhist(equalized_img);


% --- Executes on button press in pushbutton_histMatch.
function pushbutton_histMatch_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if isempty(handles.img)
    msgbox('请先加载图像', '错误', 'error');
    return;
end

% 转换为灰度图像
if size(handles.img, 3) == 3
    gray_img = rgb2gray(handles.img);
else
    gray_img = handles.img;
end

% 选择参考图像
[filename, pathname] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp'}, '选择参考图像');
if filename == 0
    return;
end
ref_img = imread(fullfile(pathname, filename));

% 转换参考图像为灰度图像
if size(ref_img, 3) == 3
    ref_gray = rgb2gray(ref_img);
else
    ref_gray = ref_img;
end

% 进行直方图匹配
matched_img = imhistmatch(gray_img, ref_gray);

% 显示匹配后的图像
axes(handles.axes_image);
imshow(matched_img);

% 显示匹配后的直方图
axes(handles.axes_histogram);
imhist(matched_img);


% --- Executes on button press in pushbutton_grayTransform.
function pushbutton_grayTransform_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if isempty(handles.img)
    msgbox('请先加载图像', '错误', 'error');
    return;
end

% 转换为灰度图像
if size(handles.img, 3) == 3
    gray_img = rgb2gray(handles.img);
else
    gray_img = handles.img;
end

% 显示灰度图像在直方图的坐标轴上
axes(handles.axes_histogram); % 使用之前显示直方图的坐标轴
imshow(gray_img);

% 保存灰度图像到 handles 结构
handles.gray_img = gray_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_linearContrast.
function pushbutton_linearContrast_Callback(hObject, eventdata, handles)
% 检查是否灰度化了图像
if ~isfield(handles, 'gray_img') || isempty(handles.gray_img)
    msgbox('请先进行灰度化操作', '错误', 'error');
    return;
end

% 获取灰度图像
gray_img = handles.gray_img;

% 线性变换公式：scale 图像像素到 [0, 255]
min_val = double(min(gray_img(:)));
max_val = double(max(gray_img(:)));
linear_img = uint8(255 * (double(gray_img) - min_val) / (max_val - min_val));

% 显示线性变换后的图像
axes(handles.axes_histogram); % 在直方图显示区域显示增强后的图像
imshow(linear_img);

% 保存增强后的图像
handles.linear_img = linear_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_logTransform.
function pushbutton_logTransform_Callback(hObject, eventdata, handles)
% 检查是否灰度化了图像
if ~isfield(handles, 'gray_img') || isempty(handles.gray_img)
    msgbox('请先进行灰度化操作', '错误', 'error');
    return;
end

% 获取灰度图像
gray_img = handles.gray_img;

% 对数变换公式：log(1 + pixel_value)
log_img = uint8(255 * mat2gray(log(1 + double(gray_img))));

% 显示对数变换后的图像
axes(handles.axes_histogram); % 在直方图显示区域显示增强后的图像
imshow(log_img);

% 保存增强后的图像
handles.log_img = log_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_expTransform.
function pushbutton_expTransform_Callback(hObject, eventdata, handles)
% 检查是否灰度化了图像
if ~isfield(handles, 'gray_img') || isempty(handles.gray_img)
    msgbox('请先进行灰度化操作', '错误', 'error');
    return;
end

% 获取灰度图像
gray_img = handles.gray_img;

% 指数变换公式：exp(pixel_value / 255) - 1
exp_img = uint8(255 * mat2gray(exp(double(gray_img) / 255) - 1));

% 显示指数变换后的图像
axes(handles.axes_histogram); % 在直方图显示区域显示增强后的图像
imshow(exp_img);

% 保存增强后的图像
handles.exp_img = exp_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_scaleTransform.
function pushbutton_scaleTransform_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if isempty(handles.img)
    msgbox('请先加载图像', '错误', 'error');
    return;
end

% 提示用户输入缩放因子
prompt = {'请输入缩放因子（例如 0.5 表示缩小 50%，2 表示放大 2 倍）：'};
dlgtitle = '输入缩放因子';
dims = [1 50];
default_input = {'1'};
answer = inputdlg(prompt, dlgtitle, dims, default_input);

% 验证用户输入
if isempty(answer)
    return; % 用户取消输入
end

scale_factor = str2double(answer{1});
if isnan(scale_factor) || scale_factor <= 0
    msgbox('请输入有效的缩放因子！', '错误', 'error');
    return;
end

% 缩放变换
scaled_img = imresize(handles.img, scale_factor);

% 显示缩放后的图像
axes(handles.axes_histogram); % 在直方图坐标轴中显示
imshow(scaled_img);

% 保存缩放后的图像
handles.scaled_img = scaled_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_rotateTransform.
function pushbutton_rotateTransform_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if isempty(handles.img)
    msgbox('请先加载图像', '错误', 'error');
    return;
end

% 提示用户输入旋转角度
prompt = {'请输入旋转角度（正值表示逆时针旋转，负值表示顺时针旋转）：'};
dlgtitle = '输入旋转角度';
dims = [1 50];
default_input = {'0'};
answer = inputdlg(prompt, dlgtitle, dims, default_input);

% 验证用户输入
if isempty(answer)
    return; % 用户取消输入
end

angle = str2double(answer{1});
if isnan(angle)
    msgbox('请输入有效的旋转角度！', '错误', 'error');
    return;
end

% 旋转变换
rotated_img = imrotate(handles.img, angle);

% 显示旋转后的图像
axes(handles.axes_histogram); % 在直方图坐标轴中显示
imshow(rotated_img);

% 保存旋转后的图像
handles.rotated_img = rotated_img;
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function slider_kernelSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_kernelSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% --- Executes during object creation, after setting all properties.
function slider_cutoffFreq_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_cutoffFreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes during object creation, after setting all properties.
function edit_kernelSize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_kernelSize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes during object creation, after setting all properties.
function edit_cutoffFreq_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_cutoffFreq (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_addNoise.
function pushbutton_addNoise_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if isempty(handles.img)
    msgbox('请先加载图像', '错误', 'error');
    return;
end

% 提示用户选择噪声类型
noise_type = questdlg('请选择噪声类型：', ...
                      '噪声类型', ...
                      '高斯噪声', '椒盐噪声', '泊松噪声', '高斯噪声');

% 根据选择添加噪声
switch noise_type
    case '高斯噪声'
        noisy_img = imnoise(handles.img, 'gaussian', 0, 0.01); % 高斯噪声
    case '椒盐噪声'
        noisy_img = imnoise(handles.img, 'salt & pepper', 0.02); % 椒盐噪声
    case '泊松噪声'
        noisy_img = imnoise(handles.img, 'poisson'); % 泊松噪声
    otherwise
        return; % 用户取消选择
end

% 显示加噪后的图像
axes(handles.axes_histogram); % 在直方图区域显示
imshow(noisy_img);

% 保存加噪后的图像
handles.noisy_img = noisy_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_applyFilter.
function pushbutton_applyFilter_Callback(hObject, eventdata, handles)
% 检查是否已加噪
if ~isfield(handles, 'noisy_img') || isempty(handles.noisy_img)
    msgbox('请先对图像加噪！', '错误', 'error');
    return;
end

% 提示用户选择滤波方式
filter_type = questdlg('请选择滤波类型：', ...
                       '滤波类型', ...
                       '空域滤波', '频域滤波', '取消', '空域滤波');

if strcmp(filter_type, '取消')
    return;
end

% 根据选择的滤波类型处理
if strcmp(filter_type, '空域滤波')
    % 空域滤波
    prompt = {'请输入滤波核大小 (3-15, 奇数)：'};
    dlgtitle = '空域滤波参数';
    dims = [1 50];
    definput = {'3'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer)
        return; % 用户取消操作
    end
    
    kernelSize = str2double(answer{1});
    if isnan(kernelSize) || kernelSize < 3 || kernelSize > 15 || mod(kernelSize, 2) == 0
        msgbox('核大小必须是3到15之间的奇数！', '错误', 'error');
        return;
    end

    % 使用均值滤波
    h = fspecial('average', [kernelSize kernelSize]);
    filtered_img = imfilter(handles.noisy_img, h, 'replicate');

elseif strcmp(filter_type, '频域滤波')
    % 频域滤波
    prompt = {'请输入截止频率 (10-100)：'};
    dlgtitle = '频域滤波参数';
    dims = [1 50];
    definput = {'50'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer)
        return; % 用户取消操作
    end
    
    cutoffFreq = str2double(answer{1});
    if isnan(cutoffFreq) || cutoffFreq < 10 || cutoffFreq > 100
        msgbox('截止频率必须在10到100之间！', '错误', 'error');
        return;
    end

    % 转换为灰度图像
    if size(handles.noisy_img, 3) == 3
        img_gray = rgb2gray(handles.noisy_img);
    else
        img_gray = handles.noisy_img;
    end

    % 进行傅里叶变换
    F = fft2(double(img_gray));
    Fshift = fftshift(F);
    [rows, cols] = size(Fshift);
    crow = round(rows / 2);
    ccol = round(cols / 2);

    % 创建低通滤波器
    H = zeros(rows, cols);
    for i = 1:rows
        for j = 1:cols
            D = sqrt((i - crow)^2 + (j - ccol)^2);
            if D <= cutoffFreq
                H(i, j) = 1;
            end
        end
    end

    % 应用滤波器
    G = Fshift .* H;
    G = ifftshift(G);
    filtered_img = real(ifft2(G)); % 逆傅里叶变换

else
    return;
end

% 显示滤波后的图像
axes(handles.axes_histogram); % 显示在直方图区域
imshow(uint8(filtered_img));

% 保存滤波后的图像
handles.filtered_img = filtered_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_edgeDetect.
function pushbutton_edgeDetect_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if ~isfield(handles, 'img') || isempty(handles.img)
    msgbox('请先加载图像！', '错误', 'error');
    return;
end

% 转换为灰度图像（如果不是灰度图）
if size(handles.img, 3) == 3
    gray_img = rgb2gray(handles.img);
else
    gray_img = handles.img;
end

% 提示用户选择边缘检测算子
operator_type = questdlg('请选择边缘检测算子：', ...
                         '边缘检测', ...
                         'Robert 算子', 'Prewitt 算子', ...
                         'Sobel 算子', 'Robert 算子');

if isempty(operator_type)
    return; % 用户取消操作
end

% 执行边缘检测
switch operator_type
    case 'Robert 算子'
        % 使用 roberts 算子
        edge_img = edge(gray_img, 'roberts');
    case 'Prewitt 算子'
        % 使用 prewitt 算子
        edge_img = edge(gray_img, 'prewitt');
    case 'Sobel 算子'
        % 使用 sobel 算子
        edge_img = edge(gray_img, 'sobel');
    otherwise
        % 拉普拉斯算子（手动实现）
        h = fspecial('laplacian', 0.2); % 拉普拉斯滤波器
        edge_img = imfilter(double(gray_img), h, 'replicate');
end

% 显示边缘检测结果
axes(handles.axes_histogram); % 在直方图区域显示
imshow(edge_img, []);

% 保存边缘检测后的图像
handles.edge_img = edge_img;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_extractBird.
function pushbutton_extractBird_Callback(hObject, eventdata, handles)
% 检查是否加载了图像
if ~isfield(handles, 'img') || isempty(handles.img)
    msgbox('请先加载图像！', '错误', 'error');
    return;
end

% 转换为灰度图像
if size(handles.img, 3) == 3
    gray_img = rgb2gray(handles.img);
else
    gray_img = handles.img;
end

% 提示用户选择分割方法
method = questdlg('请选择目标提取方法：', ...
                  '目标提取', ...
                  '边缘检测', '阈值分割', '取消', '边缘检测');

if strcmp(method, '取消')
    return;
end

% 初始化变量
bird_mask = []; % 用于存储目标区域掩码

switch method
    case '边缘检测'
        % 使用 Sobel 算子进行边缘检测
        edge_img = edge(gray_img, 'sobel');

        % 形态学操作（填充目标区域）
        se = strel('disk', 5); % 定义结构元素
        bird_mask = imclose(edge_img, se); % 闭运算填充边缘
        bird_mask = imfill(bird_mask, 'holes'); % 填充目标区域内部

    case '阈值分割'
        % 提示用户输入阈值
        prompt = {'请输入阈值 (0-255)：'};
        dlgtitle = '阈值分割';
        dims = [1 50];
        definput = {'128'};
        answer = inputdlg(prompt, dlgtitle, dims, definput);

        if isempty(answer)
            return; % 用户取消操作
        end

        threshold = str2double(answer{1});
        if isnan(threshold) || threshold < 0 || threshold > 255
            msgbox('阈值必须在 0 到 255 之间！', '错误', 'error');
            return;
        end

        % 阈值分割
        bird_mask = gray_img > threshold;

        % 形态学操作（去除小噪声）
        se = strel('disk', 3); % 定义结构元素
        bird_mask = imopen(bird_mask, se); % 开运算去噪

    otherwise
        return;
end

% 提取鸟类目标区域
bird_region = handles.img; % 使用原始图像
if size(bird_region, 3) == 3
    % 对彩色图像提取区域
    for i = 1:3
        channel = bird_region(:, :, i);
        channel(~bird_mask) = 0; % 非目标区域设为 0
        bird_region(:, :, i) = channel;
    end
else
    % 对灰度图像提取区域
    bird_region(~bird_mask) = 0; % 非目标区域设为 0
end

% 显示目标提取结果
axes(handles.axes_histogram); % 在直方图区域显示
imshow(bird_region);

% 保存目标提取后的图像
handles.bird_region = bird_region;
guidata(hObject, handles);


% --- Executes on button press in pushbutton_featureExtract.
function pushbutton_featureExtract_Callback(hObject, eventdata, handles)
% 检查是否加载了原始图像
if ~isfield(handles, 'img') || isempty(handles.img)
    msgbox('请先加载原始图像！', '错误', 'error');
    return;
end

% 检查是否提取了目标区域
if ~isfield(handles, 'bird_region') || isempty(handles.bird_region)
    msgbox('请先提取目标区域！', '错误', 'error');
    return;
end

% 提示用户选择特征提取方法
method = questdlg('请选择特征提取方法：', ...
                  '特征提取', ...
                  'LBP', 'HOG', '取消', 'LBP');

if strcmp(method, '取消')
    return;
end

% 对原始图像和目标区域进行灰度化
original_gray = rgb2gray(handles.img);
target_gray = rgb2gray(handles.bird_region);

% 初始化变量
original_feature = [];
target_feature = [];

% 提取并显示特征
switch method
    case 'LBP'
        % 调用自定义 LBP 函数
        original_feature = computeLBP(original_gray);
        target_feature = computeLBP(target_gray);

        % 显示 LBP 特征图
        figure;
        subplot(2, 2, 1);
        imshow(handles.img);
        title('原始图像');
        subplot(2, 2, 2);
        imshow(original_feature, []);
        title('原始图像的 LBP 特征图');
        
        subplot(2, 2, 3);
        imshow(handles.bird_region);
        title('目标区域');
        subplot(2, 2, 4);
        imshow(target_feature, []);
        title('目标区域的 LBP 特征图');

    case 'HOG'
        % 调用 MATLAB 的 HOG 特征提取函数
        [original_feature, original_vis] = extractHOGFeatures(original_gray);
        [target_feature, target_vis] = extractHOGFeatures(target_gray);

        % 显示 HOG 特征图
        figure;
        subplot(2, 2, 1);
        imshow(handles.img);
        title('原始图像');
        subplot(2, 2, 2);
        plot(original_vis);
        title('原始图像的 HOG 特征图');
        
        subplot(2, 2, 3);
        imshow(handles.bird_region);
        title('目标区域');
        subplot(2, 2, 4);
        plot(target_vis);
        title('目标区域的 HOG 特征图');

    otherwise
        return;
end

% 保存特征到 handles
handles.original_feature = original_feature;
handles.target_feature = target_feature;
guidata(hObject, handles);

msgbox([method ' 特征提取完成并可视化！'], '成功');

% 手动实现 LBP 特征提取的函数
function lbp_image = computeLBP(gray_img)
% 获取图像的高度和宽度
[height, width] = size(gray_img);

% 初始化输出 LBP 图像
lbp_image = zeros(height, width);

% 遍历每个像素（忽略边缘像素）
for y = 2:height-1
    for x = 2:width-1
        % 获取当前像素的 3x3 邻域
        center = gray_img(y, x);
        neighbours = [
            gray_img(y-1, x-1), gray_img(y-1, x), gray_img(y-1, x+1);
            gray_img(y, x-1),   center,          gray_img(y, x+1);
            gray_img(y+1, x-1), gray_img(y+1, x), gray_img(y+1, x+1)
        ];

        % 二值化邻域
        binary_pattern = (neighbours >= center);

        % 计算 LBP 值 (按顺序展开邻域)
        binary_pattern = binary_pattern([1, 2, 3, 6, 9, 8, 7, 4]); % 顺时针展开
        lbp_value = sum(binary_pattern .* 2.^(0:7)); % 转换为 LBP 值

        % 保存到 LBP 图像
        lbp_image(y, x) = lbp_value;
    end
end