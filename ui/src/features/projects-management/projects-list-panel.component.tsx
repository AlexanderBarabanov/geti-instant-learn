/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import {
    ActionButton,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Header,
    Heading,
    PhotoPlaceholder,
    Text,
    View,
} from '@geti/ui';
import { AddCircle } from '@geti/ui/icons';
import { useParams } from 'react-router';

import { ProjectsList } from './projects-list.component';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    name: string;
}

const SelectedProjectButton = ({ name }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet height={'max-content'} staticColor='white'>
            <View margin={'size-50'}>{name}</View>
            <View margin='size-50'>
                <PhotoPlaceholder name={name} email='' height={'size-400'} width={'size-400'} />
            </View>
        </ActionButton>
    );
};

const projects = [
    { name: 'Project 1', id: '1' },
    { name: 'Project 2', id: '2' },
    { name: 'Project 3', id: '3' },
];

export const ProjectsListPanel = () => {
    const { projectId } = useParams();

    const selectedProjectName = projects.find((project) => project.id === projectId)?.name || '';

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton name={selectedProjectName} />

            <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                <Header>
                    <Flex direction={'column'} justifyContent={'center'} width={'100%'} alignItems={'center'}>
                        <PhotoPlaceholder
                            name={selectedProjectName}
                            email=''
                            height={'size-1000'}
                            width={'size-1000'}
                        />
                        <Heading level={2} marginBottom={0}>
                            {selectedProjectName}
                        </Heading>
                    </Flex>
                </Header>
                <Content UNSAFE_className={styles.panelContent}>
                    <Divider size={'S'} marginY={'size-200'} />
                    <ProjectsList projects={projects} />
                    <Divider size={'S'} marginY={'size-200'} />
                </Content>

                <ButtonGroup UNSAFE_className={styles.panelButtons}>
                    <ActionButton
                        isQuiet
                        width={'100%'}
                        marginStart={'size-100'}
                        marginEnd={'size-350'}
                        UNSAFE_className={styles.addProjectButton}
                    >
                        <AddCircle />
                        <Text marginX='size-50'>Add project</Text>
                    </ActionButton>
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
