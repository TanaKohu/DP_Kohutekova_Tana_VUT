clc;
clear;
close;
%% Load the file
global input_image;
global trajec;
global motion
global m_scen
global output;
global output_kspace;

folders = {'C:\VUT\ING STUDIUM\ZZ.Diplomova praca\AIkorekcePohybu-Matlab\001_bez',
           'C:\VUT\ING STUDIUM\ZZ.Diplomova praca\AIkorekcePohybu-Matlab\002_bez',
           'C:\VUT\ING STUDIUM\ZZ.Diplomova praca\AIkorekcePohybu-Matlab\003_bez',
           'C:\VUT\ING STUDIUM\ZZ.Diplomova praca\AIkorekcePohybu-Matlab\004_bez',
           'C:\VUT\ING STUDIUM\ZZ.Diplomova praca\AIkorekcePohybu-Matlab\005_bez'};


%input  = {niftiread(strcat(folders{1},'\T1.nii\T1.nii')),
%         niftiread(strcat(folders{2},'\T1.nii\T1.nii')),
%         niftiread(strcat(folders{3},'\T1.nii\T1.nii')),
%         niftiread(strcat(folders{5},'\T1.nii\T1.nii')),
%         niftiread(strcat(folders{5},'\T1.nii\T1.nii'))};

input  = {load(strcat(folders{1},'\T1.nii\T1.mat')).nifti_data,
          load(strcat(folders{2},'\T1.nii\T1.mat')).nifti_data,
          load(strcat(folders{3},'\T1.nii\T1.mat')).nifti_data,
          load(strcat(folders{5},'\T1.nii\T1.mat')).nifti_data,
          load(strcat(folders{5},'\T1.nii\T1.mat')).nifti_data};

for pat = 1:size(input,1)
    input_image = input{pat};
    input_image = double(input_image);  
    %for i = 1:size(input_image,3)
    %    input_image(:,:,i) = imrotate(input_image(:,:,i), 90);  % Rotate 90 degrees to the right
    %end
    [~, ~, z] = size(input_image);
    
    assignin('base','input_image',input_image);
    [x,y,z] = size(input_image);  % display input size
    
    %% Trajection
    [x,~,z] = size(input_image);
    trajec= 1:x;
    trajec = trajec';
    
    %% Create
    % scantime information
    tr = 40;
    scantime= 0.04*256*32;
    scantime = ceil(scantime);
    
    % motion parameters
    motion_params = struct;
    
    types =  ['Periodic (Continuous)', 'Linear (Continuous)', 'Nonlinear (Continuous)', 'Sudden'];
    strength = ['Moderate', 'Severe'];
    
    motion_params.ap.type = {'X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'};
    motion_params.rl.type = {'Linear (Continuous)', 'Linear (Continuous)', 'Periodic (Continuous)', 'Periodic (Continuous)','Sudden','Sudden','X','X','X','X','X','X','X','X','X','X'};
    motion_params.is.type = {'X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'};
    motion_params.yaw.type = {'X','X','X','X','X','X', 'Sudden', 'Sudden','X','X','X','X','X','X','X','X'};
    motion_params.pitch.type = {'X','X','X','X','X','X','X','X','Sudden', 'Sudden','X','X','X','X','X','X'};
    motion_params.roll.type = {'X','X','X','X','X','X','X','X','X','X','Linear (Continuous)', 'Linear (Continuous)', 'Periodic (Continuous)', 'Periodic (Continuous)','Sudden', 'Sudden'};
    
    motion_params.ap.strength = {'X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'};
    motion_params.rl.strength = {'Moderate', 'Severe','Moderate', 'Severe','Moderate', 'Severe','X','X','X','X','X','X','X','X','X','X'};
    motion_params.is.strength = {'X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'};
    motion_params.yaw.strength = {'X','X','X','X','X','X','Moderate', 'Severe','X','X','X','X','X','X','X','X'};
    motion_params.pitch.strength = {'X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'};
    motion_params.roll.strength = {'X','X','X','X','X','X','X','X','X','X','Moderate', 'Severe','Moderate', 'Severe','Moderate', 'Severe',};
   


    for mot_index = 1:size(motion_params.ap.type,2)
        m_scen = struct;
    
        m_scen.ap.type = motion_params.ap.type{mot_index};
        m_scen.rl.type = motion_params.rl.type{mot_index};
        m_scen.is.type = motion_params.is.type{mot_index};
        m_scen.yaw.type = motion_params.yaw.type{mot_index};
        m_scen.pitch.type = motion_params.pitch.type{mot_index};
        m_scen.roll.type = motion_params.roll.type{mot_index};
        
        m_scen.ap.strength = motion_params.ap.strength{mot_index};
        m_scen.rl.strength = motion_params.rl.strength{mot_index};
        m_scen.is.strength = motion_params.is.strength{mot_index};
        m_scen.yaw.strength = motion_params.yaw.strength{mot_index};
        m_scen.pitch.strength = motion_params.pitch.strength{mot_index};
        m_scen.roll.strength = motion_params.roll.strength{mot_index};
        

        motion = motion_para(size(input_image,3), m_scen, tr, ceil(scantime));

        xq = 0:tr/1000:scantime-tr/1000;
        for i = 1:6
            motion_new(:,i) = interp1(motion(:,i),xq,'pchip');
        end
        motion = motion_new;
    
        xx = 0;
        x = scantime;
        y = max(motion,[],'all');
    
        if y > 2
            y = y + 2;
        else 
            y = 4;
        end
    
        %plot(motion(:,1))
        %xlim([xx x])
        %ylim([(-1)*y y])
        %plot(motion(:,2))
        %xlim([xx x])
        %ylim([(-1)*y y])
        %plot(motion(:,3))
        %xlim([xx x])
        %ylim([(-1)*y y])
        %plot(motion(:,4))
        %xlim([xx x])
        %ylim([(-1)*y y])
        %plot(motion(:,5))
        %xlim([xx x])
        %ylim([(-1)*y y])
        %plot(motion(:,6))
        %xlim([xx x])
        %ylim([(-1)*y y])
    
    %% Apply
        output = zeros(size(input_image));
        [ay,ax,az] = size(input_image);
        kp = zeros(size(input_image));
    
        fov_y = 256;
        fov_x = 256;
        
        res_y = 1;
        res_x = 1;
        
        if fov_y~= ay
            motion(:,1) = motion(:,1)/res_y;
            fov_y
            ay
        end
        if fov_x~= ax
            motion(:,1) = motion(:,1)/res_x;
        end
    
        %%%%%%%%%%%%%%%%%% phase encoding direction %%%%%%%%%%%%%%%%%%%%%%%
        %% setting of coding
        for zz = 1:az
            input_image_flip(:,:,zz) = imrotate(input_image(:,:,zz),0);
            kp(:,:,zz) = fft2c(input_image_flip(:,:,zz));
        end 
        
        trajma = trajec;
        for zz = 1:az
            % zero means no motion
            motma = zeros(ay,6);
            % MOTION
            motma(:,:) = motion(ay*(zz-1)+1:ay*(zz-1)+ay,:);
            
        %     zz
           % for ee = 1:ae
            ksma = kp(:,:,zz); % k-space data
            pad = 0; % no padding
        
            [ output_kdata ] = motion_simul( ksma, trajma, motma);
            output_tmp = ifft2c(output_kdata);
            output_kspace(:,:,zz) = output_kdata;
            output(:,:,zz) = output_tmp;
        end
        
        %%%%%%%%%%%%%%%%%% phase encoding direction %%%%%%%%%%%%%%%%%%%%%%%
        for i = 1:az
            output(:,:,i) = imrotate(output(:,:,i),0);
            output_kspace(:,:,i) = imrotate(output_kspace(:,:,i),0);
        end 
        
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
        [t_x,t_y]= size(motma);
        for i = 1:t_x
            if motma(i,1) ~=0
                break;
            elseif motma(i,2) ~=0
                break;
            elseif motma(i,3) ~=0
                break;
            elseif motma(i,4) ~=0
                break;
            elseif motma(i,5) ~=0
                break;
            elseif motma(i,6) ~=0
                break;
            end
        end
        za = round(i/ay);
        if za >= az
            za = az-1;
        end
    
        %% Ukladanie motion corrupted images (.mat)
        mri_data = abs(output);
        filename = strcat(folders{pat},'\T1.nii\', int2str(pat), '_motion_', int2str(mot_index), '_data.mat');
        save(filename,'mri_data');
        filename = strcat(folders{pat},'\T1.nii\', int2str(pat), '_motion_', int2str(mot_index+102), '_data.nii');
        for zz = 1:az
            mri_data_flip(:,:,zz) = imrotate(mri_data(:,:,zz),-90);% toto sa pri AP nepouziva
        end 
        niftiwrite(mri_data_flip, filename);
      
        
    
    
        %% Ukladanie motion trajektorii
        filename = strcat(folders{pat},'\T1.nii\', int2str(pat), '_motion_', int2str(mot_index+102), '_motion_params.mat');
        AP_direction = motion(:,1); 
        RL_direction = motion(:,2); 
        IS_direction = motion(:,3); 
        Yaw_direction = motion(:,4); 
        Pitch_direction = motion(:,5); 
        Roll_direction = motion(:,6); 
        save(filename,'AP_direction', 'RL_direction','IS_direction', 'Yaw_direction', 'Pitch_direction',  'Roll_direction');
    
        %% Vypocet koeficientov
        global input_image
        global output
        
        %%%%%% RMSE
        [xx,yy,zz] = size(input_image);
        
        RMSE = zeros(1,zz);
        SSIM = zeros(1,zz);
        PSNR = zeros(1,zz);
        for i = 1:zz
            mask=abs(input_image(:,:,i));mask(mask<mean(mask(:))*.5)=0;mask(mask>0)=1;
            mask=medfilt2(medfilt2(mask,[17 17]),[17 17]);
            mask = double(mask); 
        %     RMSE(i) = sqrt(sum(abs(abs(input_image(:,:,i)).*mask-abs(output(:,:,i))).*mask,'all')./(sum(mask,'all')));
        RMSE(i) = sqrt(sum(abs((input_image(:,:,i)).*mask-abs(output(:,:,i)).^2).*mask,'all')./(sum(mask,'all')));
            SSIM(i) = ssim(abs(output(:,:,i)).*mask,abs(input_image(:,:,i)).*mask);
            PSNR(i) = psnr(abs(output(:,:,i)).*mask,abs(input_image(:,:,i)).*mask);
        end

        filename = strcat(folders{pat},'\T1.nii\', int2str(pat), '_motion_', int2str(mot_index+102), '_similarity.mat');
        save(filename, 'PSNR', 'SSIM', 'RMSE');
        

        %% save kspace
        filename = strcat(folders{pat},'\T1.nii\', int2str(pat), '_motion_', int2str(mot_index+102), '_kspace.mat');
        save(filename,'output_kspace');
        disp(strcat('Hotovo pacient ', int2str(pat), ' pohyb ', int2str(mot_index)));
    end


    
end





