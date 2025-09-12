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
    Footer,
    Header,
    PhotoPlaceholder,
    Text,
    View,
} from 'packages/ui';
import { AddCircle } from 'packages/ui/icons';
import { useParams } from 'react-router';

import { ProjectsList } from './projects-list.component';

import styles from './projects-list.module.css';

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

            <Dialog width={'size-4600'}>
                <Header>
                    <Flex direction={'column'} justifyContent={'center'} width={'100%'} alignItems={'center'}>
                        <PhotoPlaceholder
                            name={selectedProjectName}
                            email=''
                            height={'size-1000'}
                            width={'size-1000'}
                        />
                        <h2>{selectedProjectName}</h2>
                    </Flex>
                </Header>
                <Content UNSAFE_className={styles.panelContent}>
                    <Divider size={'S'} margin={'size-200'} />
                    <ProjectsList projects={projects} />
                    <Divider size={'S'} margin={'size-200'} />
                </Content>
                <Footer UNSAFE_className={styles.panelFooter}>
                    <ButtonGroup>
                        <ActionButton isQuiet width={'100%'} margin={'size-200'}>
                            <AddCircle style={{ margin: 'size-100' }} />
                            <Text marginX='size-50'>Add project</Text>
                        </ActionButton>
                    </ButtonGroup>
                </Footer>
            </Dialog>
        </DialogTrigger>
    );
};
