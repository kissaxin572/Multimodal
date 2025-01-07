#!/bin/bash

# 环境变量定义
readonly LOG_PATH="/home/ubuntu20/Workspace/Datasets/Multimodal/10s"  # 结果保存路径
readonly HPC_PATH="/home/ubuntu20/Workspace/Datasets/Multimodal/10s/8Events"  # HPC结果保存路径
readonly SNAPSHOT_PATH="/home/ubuntu20/Workspace/Datasets/Multimodal/10s/Snapshots"  # SNAPSHOT结果保存路径
readonly ELF_PATH_CONTAINER="/Datasets/malwares/Valid_ELF_20200405"  # 容器内ELF文件路径
readonly ELF_PATH_LOCAL="/home/ubuntu20/Workspace/Datasets/malwares/virus"  # 本地ELF文件路径
readonly PURE_ELF_PATH="/home/ubuntu20/Workspace/Datasets/malwares/pure_Valid_ELF_20200405"  # 纯净ELF文件路径
readonly BENIGN_PATH="/home/ubuntu20/Workspace/Datasets/benign/Benign_Software.txt"  # 良性软件列表路径
readonly IMAGE_NAME="pure_ubuntu20"  # Docker镜像名称
readonly CONTAINER_NAME="base"  # Docker容器名称

# 定义需要收集的所有硬件性能计数器事件数组
readonly EVENTS="branch-instructions,branch-misses,cache-misses,cpu-cycles,instructions,L1-dcache-loads,LLC-stores,iTLB-load-misses"

# 日志输出函数,输出带时间戳的日志信息
log_msg() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_PATH/log.txt
}

# 文件索引计数器,用于生成唯一的输出文件名
file_index=1

# 从纯净ELF文件目录获取所有恶意软件样本文件列表
mapfile -t elf_files < <(ls "$PURE_ELF_PATH")
log_msg "已获取ELF文件列表"

# 从文本文件中读取良性软件列表
benign_files=()
while IFS= read -r line || [[ -n "$line" ]]; do
    benign_files+=("$line")
done < "$BENIGN_PATH"
log_msg "已获取良性软件列表"

# 删除Docker容器的函数,停止并删除指定名称的容器
delete_container() {
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    log_msg "已删除容器 ${CONTAINER_NAME}"
}

# 在Docker容器中执行文件并收集性能计数器以及快照数据的核心函数
get_data() {
    local hpc_result="$1"
    local snapshot_result="$2"
    local exec_file="$3"     # 要在容器中执行的文件路径
    local duration_time="10s"  # 执行和采集数据的持续时间(秒)
    
    # 创建新的Docker容器,挂载本地恶意软件目录到容器内
    if ! docker run -d --name "$CONTAINER_NAME" \
        --memory=4g \
        --cpu-shares=1024 \
        --privileged \
        --security-opt seccomp:unconfined \
        -v "${ELF_PATH_LOCAL}:${ELF_PATH_CONTAINER}" \
        "$IMAGE_NAME" \
        /bin/bash -c "$exec_file; tail -f /dev/null" ; then
        log_msg "错误: 无法创建容器 ${CONTAINER_NAME}"
        exit 1
    fi
    log_msg "已创建容器 ${CONTAINER_NAME}"
    
    # 获取新创建容器的ID,用于perf命令中的容器标识
    local container_id
    container_id=$(docker inspect --format '{{.Id}}' "$CONTAINER_NAME")
    log_msg "容器 ${CONTAINER_NAME} 的ID为 ${container_id}"
    
    # 构建perf命令的参数,为每个性能计数器事件添加容器ID前缀
    local perf_cmd
    local group_args
    group_args=$(printf ',docker/%s' $(for i in {1..8}; do echo "$container_id"; done))
    group_args=${group_args:1}  # 移除开头的逗号

    # 构建perf命令
    perf_cmd="timeout --signal=SIGINT ${duration_time} perf stat -e $EVENTS -G $group_args -o $hpc_result"
    log_msg "执行命令: $perf_cmd"
    eval "$perf_cmd &"  # 在后台执行perf命令,开始收集性能计数器数据
    log_msg "等待 ${duration_time}秒"
    
    local perf_pid=$!  # 记录perf命令的进程ID
    # 等待perf命令执行完成,即等待数据收集结束
    wait "$perf_pid"

    # 不暂停容器的情况下创建检查点
    if ! docker checkpoint create --leave-running=true "$CONTAINER_NAME" "ck1"; then
        log_msg "错误: 无法创建检查点 ck1 for 容器 ${CONTAINER_NAME}"
        criu_log_path="/run/containerd/io.containerd.runtime.v2.task/moby/${container_id}/criu-dump.log"
        if [ -f "$criu_log_path" ]; then
            log_msg "CRIU 日志内容:"
            cat "$criu_log_path" >> "$LOG_PATH/log.txt"
        fi
        
        # 将无法创建检查点的exec_file保存到failure.txt文件中
        echo "${exec_file}" >> "$LOG_PATH/failure_10s.txt"
        
        # 无法创建检查点时，跳转到下一个文件
        break
    fi
    
    # 保存快照文件
    cp_command="cp -r \"/var/lib/docker/containers/${container_id}/checkpoints/ck1\" \"${snapshot_result}/ck1\""
    eval "$cp_command"
    if [ $? -ne 0 ]; then
        log_msg "警告: 无法复制检查点 ${checkpoint_prefix}${i} 到 ${output_dir}"
    fi
    
    ((file_index++))

    
    # 清理:删除使用过的容器
    delete_container
}

# 运行恶意软件样本并收集数据的函数
run_ELF() {
    file_index=1
    
    # 遍历所有恶意软件样本
    for file in "${elf_files[@]}"; do
        # 将样本从纯净目录复制到本地目录
        if ! cp "${PURE_ELF_PATH}/${file}" "${ELF_PATH_LOCAL}/${file}"; then
            log_msg "错误: 无法复制ELF文件 ${file} 从 ${PURE_ELF_PATH} 到 ${ELF_PATH_LOCAL}"
            exit 1
        fi
        log_msg "已将ELF文件 ${file} 从 ${PURE_ELF_PATH} 复制到 ${ELF_PATH_LOCAL}"
        
        local hpc_result="${HPC_PATH}/M_${file_index}.txt"
        local snapshot_result="${SNAPSHOT_PATH}/M_${file_index}"
        log_msg "开始运行恶意软件 ${file}. 结果保存路径:${hpc_result} && ${snapshot_result}"
        
        # 创建结果保存文件夹
        mkdir -p "${snapshot_result}"
        
        # 执行样本并收集数据
        if ! get_data "${hpc_result}" "${snapshot_result}" "${ELF_PATH_CONTAINER}/$file"; then
            log_msg "错误: 在处理文件 ${file} 时出错"
        fi
        
        # 清理:删除本地样本文件
        rm -f "${ELF_PATH_LOCAL}/${file}"
        if [ $? -eq 0 ]; then
            log_msg "已从 ${ELF_PATH_LOCAL} 删除ELF文件 ${file}"
        else
            log_msg "警告: 无法删除 ${ELF_PATH_LOCAL}/${file}"
        fi
    done
}

# 运行良性软件样本并收集数据的函数
run_benign() {
    file_index=1
    
    # 遍历所有良性软件命令
    for file in "${benign_files[@]}"; do
        local hpc_result="${HPC_PATH}/B_${file_index}.txt"
        local snapshot_result="${SNAPSHOT_PATH}/B_${file_index}"
        log_msg "开始运行良性软件 ${file}. 保存路径:${hpc_result} && ${snapshot_result}"

        # 创建结果保存文件夹
        mkdir -p "${snapshot_result}"
        
        # 执行样本并收集数据
        if ! get_data "${hpc_result}" "${snapshot_result}" "${file}"; then
            log_msg "错误: 在处理良性软件 ${file} 时出错"
        fi
    done
}

# 执行一次完整实验的函数,包括运行所有恶意和良性软件样本
run_experiment() {    
    # 依次运行恶意和良性软件样本
    run_ELF
    run_benign
}

# 主函数:执行四组不同参数的实验
main() {
    # 禁用NMI watchdog以避免干扰性能计数器
    echo 0 > /proc/sys/kernel/nmi_watchdog

    # 创建本地临时目录用于存放样本文件
    mkdir -p "$ELF_PATH_LOCAL"
    mkdir -p "$LOG_PATH"
    mkdir -p "$HPC_PATH" # HPC结果保存路径
    mkdir -p "$SNAPSHOT_PATH" # SNAPSHOT结果保存路径

    run_experiment
}


# 调用主函数
main
